# Controle Inteligente do Pêndulo Invertido

Este projeto explora a aplicação de diferentes técnicas de inteligência computacional para o controle de um pêndulo invertido montado sobre um carrinho. O objetivo é manter o pêndulo na posição vertical e controlar a posição do carrinho dentro de limites desejados. Três abordagens de controle foram implementadas e comparadas: um Sistema de Inferência Fuzzy (FIS), um controlador Genético-Fuzzy e um controlador Neuro-Fuzzy baseado em uma Rede Neural Artificial MLP (Multi-layer Perceptron).

## Conceitos Envolvidos

### Algoritmos Genéticos (GAs)

Algoritmos Genéticos são algoritmos de otimização metaheurísticos inspirados no processo de seleção natural, um mecanismo da evolução biológica. Eles são usados para encontrar soluções aproximadas para problemas de otimização e busca. Um GA envolve uma população de potenciais soluções (indivíduos ou cromossomos), que passam por um processo iterativo de seleção (baseado em uma função de fitness que avalia a qualidade da solução), cruzamento (combinação de informações de dois pais para gerar novos descendentes) e mutação (pequenas alterações aleatórias em um cromossomo) para evoluir ao longo de gerações em direção a soluções melhores.

### Sistema de Inferência Fuzzy (FIS)

Um Sistema de Inferência Fuzzy é um sistema de raciocínio baseado na lógica fuzzy, que lida com o conceito de "grau de verdade" em vez da lógica booleana tradicional (verdadeiro ou falso). Um FIS tipicamente consiste em quatro componentes principais:

1.  **Base de Regras:** Um conjunto de regras IF-THEN que definem o comportamento do sistema em termos de variáveis linguísticas (por exemplo, "SE o ângulo é Pequeno E a velocidade angular é Média ENTÃO a força é Positiva").
2.  **Fuzzificação:** O processo de mapear valores de entrada nítidos (crisp values) para graus de pertinência em conjuntos fuzzy linguísticos (por exemplo, um ângulo de 0.1 radianos pode ter um grau de 0.8 em "Pequeno" e 0.2 em "Médio").
3.  **Mecanismo de Inferência:** O processo de aplicar os graus de pertinência das entradas às regras fuzzy para determinar os graus de pertinência dos conjuntos fuzzy de saída.
4.  **Defuzzificação:** O processo de converter os conjuntos fuzzy de saída resultantes de volta para um valor de saída nítido (crisp value) que pode ser usado para controlar o sistema (por exemplo, calcular um valor de força específico a partir de um conjunto fuzzy de força).

### Rede Neural Artificial MLP (Multi-layer Perceptron)

Uma Rede Neural Artificial é um modelo computacional inspirado na estrutura e função do sistema nervoso biológico. Um Perceptron de Múltiplas Camadas (MLP) é um tipo de rede neural feedforward composta por múltiplas camadas de nós (neurônios): uma camada de entrada, uma ou mais camadas escondidas e uma camada de saída. Cada nó em uma camada está conectado a todos os nós na camada seguinte, com cada conexão tendo um peso associado. Os neurônios aplicam uma função de ativação (linear ou não linear) à soma ponderada de suas entradas para produzir sua saída. MLPs são capazes de aprender relações complexas entre entradas e saídas através de um processo de treinamento, onde os pesos das conexões são ajustados iterativamente com base em um conjunto de dados de treinamento para minimizar um erro entre as saídas previstas e os valores alvo.

## Resultados Apresentados

Os resultados das simulações iniciais com as três metodologias (Fuzzy Controller, Genético-Fuzzy e Neuro-Fuzzy) mostram que, sob as configurações atuais, o pêndulo tende a cair ao longo do tempo. Isso é evidente pelo aumento do erro angular e pela falta de estabilização na visualização 3D e nos gráficos de estado.

* **Fuzzy Controller:** Aplica um conjunto de regras fuzzy predefinidas para controlar a força aplicada ao carrinho. Os resultados indicam que essas regras, na sua forma atual, não são suficientes para manter o pêndulo estável nas condições iniciais testadas.
* **Genético-Fuzzy:** Utiliza um algoritmo genético para otimizar os parâmetros do controlador fuzzy. Apesar do processo de otimização, os resultados da simulação ainda mostram a queda do pêndulo, sugerindo que o espaço de busca pode precisar ser melhor definido, a função de fitness ajustada ou o número de gerações aumentado.
* **Neuro-Fuzzy:** Emprega uma rede neural MLP treinada com dados gerados pelo controlador fuzzy para aprender a função de controle. Os resultados atuais indicam que, apesar do treinamento, o controlador Neuro-Fuzzy também não conseguiu estabilizar o pêndulo. Isso pode ser devido a diversos fatores, incluindo a qualidade e quantidade dos dados de treinamento, a arquitetura da rede neural, os hiperparâmetros de treinamento ou a forma como a rede está sendo utilizada para predição da força de controle.

## Resultados da Simulação

As simulações realizadas com os três algoritmos de controle nas condições iniciais testadas resultaram na queda do pêndulo invertido. A análise dos gráficos de erro angular, posição e velocidade do carro, velocidade angular do pêndulo e força de controle para cada método revela o seguinte:
OBS: Imagens dos gráficos com os resultados dos algoritmos estão na pasta `/images`.

### 1. Fuzzy Inference System (FIS)

O FIS aplicou uma força de controle baseada em regras fuzzy predefinidas. No entanto, essa força não foi suficiente para contrabalancear a dinâmica do sistema, levando a um aumento progressivo do erro angular e à queda do pêndulo. A posição e a velocidade do carro oscilaram em resposta ao movimento do pêndulo, mas sem sucesso na estabilização.

### 2. Genético-Fuzzy

O controlador genético-fuzzy, cujos parâmetros foram otimizados por um algoritmo genético, também não conseguiu estabilizar o pêndulo. Embora o algoritmo genético tenha buscado melhorar o desempenho do controlador fuzzy, a solução encontrada ainda resultou na perda de equilíbrio do pêndulo. Isso sugere que a função de fitness, os limites de otimização ou o número de gerações do algoritmo genético podem precisar de ajustes.

### 3. Neuro-Fuzzy (MLP)

O controlador Neuro-Fuzzy, baseado em uma rede neural MLP treinada com dados gerados pelo FIS, também falhou em manter o pêndulo na posição vertical. A rede neural aprendeu uma função de controle que, nas condições testadas, não foi eficaz para a estabilização. Isso pode ser atribuído à qualidade ou quantidade dos dados de treinamento, à arquitetura da rede neural ou aos hiperparâmetros de treinamento.

## Diferenças entre as Metodologias
```
| Metodologia      | Pontos Fortes                                                                 | Pontos Fracos                                                                                                | Cenários Mais Adequados                                                                                             |
| :--------------- | :---------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------ |
| **Fuzzy Control** | Intuitivo, fácil de interpretar (baseado em conhecimento humano), robusto a ruído. | Difícil de otimizar para desempenho ideal, pode exigir ajuste manual extenso, escalabilidade limitada para sistemas complexos. | Sistemas com conhecimento especializado disponível, prototipagem rápida, requisitos de interpretabilidade elevados. |
| **Genético-Fuzzy** | Capacidade de otimizar automaticamente os parâmetros do controlador fuzzy, melhor desempenho potencial em comparação com o FIS manual. | Complexidade na definição da função de fitness e dos parâmetros do GA, convergência não garantida, interpretabilidade reduzida após otimização. | Sistemas onde o ajuste manual é difícil ou demorado, busca por melhor desempenho sem conhecimento especializado detalhado. |
| **Neuro-Fuzzy** | Capacidade de aprender relações complexas a partir de dados, adaptabilidade a mudanças e incertezas, potencial para alto desempenho. | Requer grandes quantidades de dados de treinamento de qualidade, "caixa preta" (interpretabilidade limitada), desempenho dependente da arquitetura e hiperparâmetros da rede. | Sistemas complexos com muitos dados disponíveis, requisitos de adaptabilidade e desempenho elevado, interpretabilidade menos crítica. |
```
## Comparação das Três Soluções
```
| Critério                 | FIS                                                                 | Genético-Fuzzy                                                                    | Neuro-Fuzzy (MLP)                                                                 |
| :----------------------- | :------------------------------------------------------------------ | :-------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- |
| **Precisão do Controle** | Baixa (não estabilizou o pêndulo nas condições testadas)           | Baixa (não estabilizou o pêndulo nas condições testadas)                          | Baixa (não estabilizou o pêndulo nas condições testadas)                          |
| **Adaptabilidade** | Limitada (depende das regras predefinidas)                          | Potencialmente maior (parâmetros podem ser otimizados)                            | Potencialmente alta (pode aprender a partir de dados)                            |
| **Eficiência (Tempo)** | Geralmente rápido (cálculo direto baseado em regras)                 | Mais lento (requer otimização offline)                                            | Rápido após o treinamento (inferência eficiente)                                  |
| **Lidar com Incertezas** | Robusto a ruído (característica da lógica fuzzy)                     | Pode melhorar a robustez através da otimização                                    | Potencialmente robusto se treinado com dados que incluam incertezas               |
| **Interpretabilidade** | Alta (regras fuzzy são compreensíveis)                              | Média (parâmetros otimizados podem ser menos intuitivos)                           | Baixa ("caixa preta")                                                             |
```
**Vantagens e Desvantagens:**

* **FIS:**
    * **Vantagens:** Intuitivo, fácil de implementar inicialmente, bom para prototipagem rápida onde o conhecimento do sistema pode ser codificado em regras.
    * **Desvantagens:** Desempenho limitado se as regras não forem bem ajustadas, difícil de otimizar para sistemas complexos.
* **Genético-Fuzzy:**
    * **Vantagens:** Capacidade de otimizar automaticamente os parâmetros do controlador fuzzy, potencialmente melhor desempenho que o FIS manual.
    * **Desvantagens:** Complexidade na definição da função de fitness, processo de otimização computacionalmente intensivo, resultados da otimização podem ser difíceis de interpretar.
* **Neuro-Fuzzy:**
    * **Vantagens:** Pode aprender relações complexas a partir de dados, alta adaptabilidade potencial, bom desempenho para sistemas não lineares.
    * **Desvantagens:** Requer grandes quantidades de dados de treinamento de qualidade, interpretabilidade limitada, desempenho fortemente dependente da arquitetura da rede e dos hiperparâmetros de treinamento.

**Cenários Específicos:**

* **FIS:** Adequado para sistemas com conhecimento especializado bem definido e onde a interpretabilidade é crucial.
* **Genético-Fuzzy:** Útil quando o ajuste manual de um FIS é difícil e busca-se um desempenho melhorado através da otimização automática.
* **Neuro-Fuzzy:** Preferível para sistemas complexos com muitos dados disponíveis, onde a adaptabilidade e o alto desempenho são prioritários, e a interpretabilidade é menos crítica.

## Conclusão e Próximos Passos

Os resultados atuais indicam que as configurações dos controladores e/ou o processo de treinamento do Neuro-Fuzzy precisam ser revisados e ajustados para alcançar o controle estável do pêndulo invertido. Os próximos passos devem incluir:

* **Análise Detalhada:** Investigar os parâmetros específicos de cada controlador e a dinâmica da simulação.
* **Ajuste do FIS:** Refinar as regras e os conjuntos fuzzy com base em princípios de controle e experimentação.
* **Otimização Genética:** Ajustar a função de fitness e os parâmetros do GA para garantir uma busca eficaz no espaço de soluções.
* **Melhoria do Neuro-Fuzzy:** Aumentar e diversificar os dados de treinamento, experimentar com diferentes arquiteturas de rede e hiperparâmetros de treinamento, e implementar corretamente a lógica de predição da força baseada nas ativações das regras fuzzy.

Ao abordar essas áreas, espera-se que as soluções de controle inteligentes demonstrem a capacidade de estabilizar o pêndulo invertido e controlar a posição do carrinho de forma eficaz.

As simulações iniciais indicam que nenhum dos três controladores, com as configurações atuais, conseguiu estabilizar o pêndulo invertido. É crucial revisar e ajustar os parâmetros, a lógica e os processos de treinamento de cada abordagem. Os próximos passos devem focar em:

* **Refinamento do FIS:** Ajustar as regras e conjuntos fuzzy com base em uma análise mais aprofundada da dinâmica do sistema.
* **Otimização Genética Aprimorada:** Experimentar com diferentes funções de fitness e aumentar o número de gerações do algoritmo genético.
* **Melhoria do Neuro-Fuzzy:** Coletar mais dados de treinamento, possivelmente explorando diferentes políticas de controle para gerar os dados, ajustar a arquitetura da rede neural e os hiperparâmetros de treinamento, e garantir a correta implementação da fase de predição.

Através dessas melhorias, espera-se que ao menos um dos controladores demonstre a capacidade de estabilizar o pêndulo invertido de forma eficaz.