# Monitoramento de Vagas com OpenCV

Este projeto utiliza Python e OpenCV para detectar objetos em vagas de estacionamento usando um modelo de detecção treinado (MobileNetSSD). Ele exibe em tempo real se a vaga está ocupada e a cor predominante do objeto.

## 📊 Funcionalidades

* Detecção de carros, pessoas, bicicletas e motos
* Estimativa da cor predominante do objeto na vaga
* Subtração de fundo para identificar ocupações que o modelo não reconhece
* Interface de visualização em tempo real

## 📊 Requisitos

Instale as dependências com:

```bash
pip install -r requirements.txt
```

## 🗂 Arquivos necessários

* `verifica_vaga.py`: script principal
* `Estacionamento.mp4`: vídeo com as vagas
* `MobileNetSSD_deploy.prototxt`: descrição da rede
* `MobileNetSSD_deploy.caffemodel`: pesos treinados da rede

## ▶️ Como executar

1. Coloque todos os arquivos na mesma pasta
2. Execute o script:

```bash
python3 verifica_vaga.py
```

3. Pressione `q` ou `Esc` para sair da execução

## 📄 Estrutura sugerida

```
verifica_vaga/
├── Estacionamento.mp4
├── MobileNetSSD_deploy.caffemodel
├── MobileNetSSD_deploy.prototxt
├── requirements.txt
├── verifica_vaga.py
└── README.md
```

## 👨‍💻 Autor

Daniel Santos, Luiz Fernando, Rafael e Gustavo Müller — Projeto de Computação Gráfica
