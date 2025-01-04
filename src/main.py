import tensorflow as tf
from keras.api.models import Model
from keras.api.layers import Dense, GlobalAveragePooling2D
from keras.api.datasets import cifar10
from keras.api.utils import to_categorical


def main():
    """
    Executa o pipeline de Transfer Learning utilizando o MobileNetV2 pré-treinado no dataset CIFAR-10.

    Passos:
    1. Carrega o dataset CIFAR-10.
    2. Preprocessa os dados (normalização e codificação one-hot).
    3. Cria o modelo base com MobileNetV2 pré-treinado.
    4. Adiciona camadas personalizadas ao modelo.
    5. Congela o modelo base para preservar pesos.
    6. Treina o modelo final nas novas camadas adicionadas.
    7. Avalia o desempenho no conjunto de teste.
    8. Salva o modelo ajustado em um arquivo .h5.
    """

    # 1. Carregar o conjunto de dados CIFAR-10
    """
    O dataset CIFAR-10 é um conjunto de dados amplamente utilizado na pesquisa de aprendizado de máquina. 
    Ele contém 60.000 imagens coloridas divididas em 10 classes (aviões, carros, pássaros, gatos, etc.). 
    As imagens são pequenas, com dimensões de 32x32 pixels, o que facilita o treinamento de modelos em computadores convencionais.
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # 2. Normalizar os dados
    """
    As imagens no dataset CIFAR-10 possuem valores de pixel que variam de 0 a 255 (escala de cores RGB). 
    Para facilitar o treinamento, esses valores são normalizados para o intervalo [0, 1], dividindo por 255.
    Isso ajuda o modelo a convergir mais rápido durante o treinamento, pois mantém os valores em uma escala menor e mais consistente.
    """
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # 3. Converter as classes para formato one-hot encoding
    """
    No CIFAR-10, as classes são representadas por números inteiros entre 0 e 9. Por exemplo:
    - Classe 0: Avião
    - Classe 1: Automóvel

    O formato one-hot encoding transforma cada número em um vetor binário com 10 posições. Exemplo:
    - Classe 3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    Isso é necessário porque a última camada do modelo usa a função softmax para calcular probabilidades para cada classe.
    """
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # 4. Criar o modelo base pré-treinado MobileNetV2
    """
    MobileNetV2 é uma arquitetura de rede neural eficiente projetada para dispositivos móveis e de baixa potência.
    Ele foi treinado no ImageNet, um grande conjunto de dados com milhões de imagens em milhares de categorias.

    Aqui, usamos o MobileNetV2 como modelo base e reutilizamos seus pesos para extrair recursos visuais das imagens do CIFAR-10.
    - weights="imagenet": Indica que queremos usar os pesos pré-treinados do ImageNet.
    - include_top=False: Remove a camada de classificação original (porque o CIFAR-10 tem 10 classes, não as milhares do ImageNet).
    - input_shape=(32, 32, 3): Define o formato das imagens de entrada, compatível com o CIFAR-10.
    """
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",  # Pesos pré-treinados
        include_top=False,  # Remove a última camada
        input_shape=(32, 32, 3)  # Formato das imagens do CIFAR-10
    )

    # 5. Congelar as camadas do modelo base
    """
    Congelar as camadas significa impedir que os pesos do modelo base sejam atualizados durante o treinamento.
    Isso preserva os conhecimentos do MobileNetV2 adquiridos no ImageNet, enquanto treinamos apenas as novas camadas adicionadas.
    """
    for layer in base_model.layers:
        layer.trainable = False

    # 6. Adicionar camadas customizadas ao modelo
    """
    Adicionamos novas camadas ao modelo para ajustar sua saída ao CIFAR-10:
    - GlobalAveragePooling2D: Reduz a dimensionalidade ao calcular a média global de cada canal.
    - Dense(128): Adiciona uma camada totalmente conectada com 128 neurônios e ativação ReLU (para não linearidades).
    - Dense(10): Camada final com 10 neurônios (um para cada classe do CIFAR-10), usando ativação softmax para produzir probabilidades.
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    predictions = Dense(10, activation="softmax")(x)

    # 7. Criar o modelo final combinando o MobileNetV2 com as camadas personalizadas
    """
    O modelo final combina as camadas do MobileNetV2 (modelo base) com as novas camadas personalizadas.
    A entrada será a mesma do MobileNetV2, e a saída será as previsões do CIFAR-10.
    """
    model = Model(inputs=base_model.input, outputs=predictions)

    # 8. Compilar o modelo
    """
    Compilamos o modelo para prepará-lo para o treinamento:
    - optimizer="adam": Um otimizador adaptativo que ajusta os pesos durante o treinamento.
    - loss="categorical_crossentropy": Função de perda para classificação multiclasse.
    - metrics=["accuracy"]: Acurácia é usada como métrica de avaliação.
    """
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # 9. Treinar o modelo
    """
    Durante o treinamento, o modelo ajusta os pesos das novas camadas para minimizar a perda.
    - epochs=5: O modelo verá todo o conjunto de treinamento 5 vezes.
    - batch_size=64: Processa 64 imagens por vez antes de atualizar os pesos.
    """
    model.fit(
        X_train, y_train,  # Dados de treinamento
        validation_data=(X_test, y_test),  # Dados de validação
        epochs=5,
        batch_size=64
    )

    # 10. Avaliar o modelo no conjunto de teste
    """
    Avaliamos o desempenho do modelo no conjunto de teste (dados não vistos durante o treinamento):
    - loss: Calcula a perda média no conjunto de teste.
    - accuracy: Mede a porcentagem de previsões corretas.
    """
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Loss: {loss}, Accuracy: {accuracy}")

    # 11. Salvar o modelo treinado
    """
    Salvamos o modelo ajustado em um arquivo .h5, que pode ser carregado posteriormente para inferência ou mais treinamento.
    """
    model.save("cifar10_mobilenetv2_finetuned.h5")
    print("Modelo salvo como 'cifar10_mobilenetv2_finetuned.h5'.")


if __name__ == "__main__":
    main()
