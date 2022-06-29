# Gerador de moléculas e preditor das suas capacidades antioxidantes
### (Projeto Final da Licenciatura em Bioquímica)
## Objetivos
ETC ETC

## Descrição Técnica
### Pastas neste projeto
#### ./raw_data/ (possivelmente não disponível no github para gestão de espaço)
Aqui estão armazenados todos os ficheiros de dados extraídos diretamente da base de dados. Não têm qualquer tipo de processamento exceto aquele aplicado na origem, como filtros de pesquisa e afins.
#### ./datasets/
Aqui estão armazenados os datasets (class: tf.data.Dataset) processados. É para aqui que está direcionado o output de todos os dataset_reader.py. São objetos guardados com pickle.dump(), pelo que podem ser carregados com pickle.load()
#### ./gerador/
Estão aqui todos os scripts relacionados com o Gerador desenvolvido neste projeto.