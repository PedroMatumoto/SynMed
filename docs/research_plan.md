# 📌 Sistema de Apoio à Identificação de Efeitos Colaterais de Medicamentos a partir de Sintomas Relatados por Usuários

## Área temática
**Saúde digital, farmacovigilância**

---

## Problema ou pergunta de pesquisa
Como identificar de forma rápida e confiável se os sintomas relatados por um paciente após a ingestão de um medicamento correspondem a efeitos colaterais conhecidos, e se possivelmente ele pode ter alguma contraindicação para aquilo.

---

## Justificativa
A automedicação e a falta de acompanhamento médico imediato podem levar ao agravamento de reações adversas a medicamentos.  
Um sistema capaz de auxiliar na detecção de possíveis efeitos colaterais pode apoiar tanto pacientes quanto profissionais de saúde na tomada de decisão.  
Além disso, contribui para a farmacovigilância, reduzindo riscos à saúde pública e fortalecendo a segurança do uso de medicamentos.

---

## Objetivo geral
Desenvolver um sistema capaz de analisar sintomas relatados por usuários após o uso de medicamentos e identificar se estes estão relacionados a efeitos colaterais documentados.

---

## Objetivos específicos
- Concatenar e relacionar bases de dados com informações sobre medicamentos e seus efeitos adversos.  
- Estruturar um modelo de análise capaz de correlacionar sintomas relatados com efeitos colaterais conhecidos.  
- Desenvolver uma interface para entrada dos sintomas e nome do medicamento.  
- Avaliar a precisão do sistema em diferentes cenários de sintomas múltiplos.  
- Fornecer recomendações iniciais de ação, como buscar atendimento médico em casos de maior gravidade.  

---

## Fontes e tipo de dados
- **Bases públicas de dados sobre medicamentos e reações adversas**:  
  - [SIDER - Side Effect Resource](http://sideeffects.embl.de/download/)  
  - [Kaggle - Drug Labels and Side Effects](https://www.kaggle.com/datasets/pratyushpuri/drug-labels-and-side-effects-dataset-1400-records)  
  - [Dados.gov.br - Medicamentos registrados no Brasil](https://dados.gov.br/dados/conjuntos-dados/medicamentos-registrados-no-brasil)  

- **Formato**: tabelas estruturadas em TSV/CSV.  
- **Abrangência**: medicamentos de uso comum, reações adversas mais frequentes e graves.  

---

## Metodologia inicial
- Extração e limpeza de dados de bancos de farmacovigilância.  
- Criação de um modelo de **Processamento de Linguagem Natural (PLN)** para interpretar sintomas em linguagem natural ou utilização de modelos existentes (ex.: GPT OSS, Ollama).  
- Implementação de um sistema de classificação/checagem utilizando:  
  - correspondência de similaridade de embeddings semânticos;  
  - classificação de conjuntos de sintomas relacionados a um medicamento.  
- Utilizar uma **IA generativa** para retornar a resposta de maneira mais amigável.  

---

## Resultados esperados
- Um protótipo funcional de sistema capaz de receber sintomas relatados pelo usuário e verificar sua compatibilidade com efeitos colaterais conhecidos de determinado medicamento.  
- Contribuição para maior segurança no uso de medicamentos, auxiliando pacientes e profissionais da saúde.  
- Democratização do conhecimento sobre medicações e maior transparência para o paciente saber o que exatamente o remédio pode gerar.  

---

## Possíveis desafios e limitações
- Sintomas vagos ou descritos de forma diferente pelo usuário (ex.: "tontura" vs "vertigem").  
- Incompletude das bases de dados de efeitos colaterais (ausência de correspondência 1:1).  
- Diferença entre efeitos colaterais comuns e graves, exigindo **regras de priorização** (bases não possuem nível de gravidade).  
- Limitação de escopo inicial (nem todos os medicamentos poderão ser contemplados de imediato).  
- Dependência de validação clínica — o sistema **não substitui diagnóstico médico**, sendo necessário explicitar essa regra na resposta gerada.  
