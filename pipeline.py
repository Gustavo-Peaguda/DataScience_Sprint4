"""pipeline.py — Sprint 4, Data Science & Statistical Computing (FIAP Challenge 2026 / TOTVS).

Fluxo: Transcrição -> Processamento -> Modelo -> Indicadores -> Recomendação.

Este módulo reúne TUDO o que não é interface: base sintética, limpeza de texto, divisão de dados,
treinamento/avaliação, artefato, indicadores por regras, scoring e recomendações. O notebook e o
`app.py` apenas importam as funções daqui.

Linha de comando (veja o README):
    python pipeline.py gerar      # gera a base SINTÉTICA reunioes_transcricoes_mockado.csv
    python pipeline.py treinar    # treina, compara modelos e salva modelo_risco.joblib
    python pipeline.py avaliar    # reavalia candidatos e confere o artefato salvo no mesmo hold-out
    python pipeline.py testar     # teste de fumaça do artefato e da inferência
    python pipeline.py analisar "texto da transcrição"

AVISO: a base é SINTÉTICA. Métricas obtidas nela demonstram que o código funciona; NÃO representam
desempenho em dados reais da TOTVS nem em cancelamentos reais de clientes.
"""
import argparse
import functools
import json
import random
import re
import sys
import tempfile
import unicodedata
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

# ============================================================================ constantes
PASTA = Path(__file__).resolve().parent
JSON_PADRAO = PASTA / "ANON_transcricao.json"            # dados ORIGINAIS (JSON Lines, 1 registro por linha)
CSV_PADRAO = PASTA / "reunioes_transcricoes_mockado.csv"  # CSV original "mockado": 4 textos únicos (não usado no treino)
SINTETICO_PADRAO = PASTA / "base_sintetica_demo.csv"      # saída de `pipeline.py gerar` (demonstração, não entregue)
ARTEFATO_PADRAO = PASTA / "modelo_risco.joblib"

RANDOM_STATE = 42
COLUNA_TEXTO = "ANON_TRANSCRICAO"
COLUNA_NPS = "NOTA_NPS"
LIMIAR_NPS = 7                  # ALVO_RISCO = (NOTA_NPS < 7)  -> risco de INSATISFAÇÃO, não churn real
MIN_CHARS = 20                  # transcrição mais curta que isso é rejeitada
MAX_CHARS = 300_000             # proteção contra entradas gigantes (transcrições reais chegam a ~175 mil)
MIN_TOKENS = 3                  # tokens úteis após a limpeza
MIN_GRUPOS_UNICOS = 100         # abaixo disso, agrupar por cliente/texto idêntico degenera o split
TEST_SIZE_FOLDS = 4             # 1 fold de 4 = 25% de teste
ESPECIFICIDADE_MIN = 0.50       # política do limiar: liberar >= 50% dos "seguros" (decidida só no treino)
AUC_CV_MIN_ELEGIVEL = 0.60      # critérios fixados ANTES de olhar o teste
AUC_MIN_SATISFATORIO = 0.65
BAL_ACC_MIN_SATISFATORIO = 0.60
JANELA_LIMIAR_SCORE = 5.0      # "score próximo ao limiar": a menos de 5 pontos do limiar
PREVISTO_RISCO_MAX = 0.95       # elegibilidade: quem prevê risco para quase todos é descartado
COBERTURA_MIN_FRACAO = 0.25     # texto com menos de 25% da MEDIANA de termos TF-IDF por transcrição do treino é "inconclusivo"

STATUS_DEMO = "demonstracao_tecnica_base_sintetica"
STATUS_REAL = "experimental_dados_originais"   # critérios mínimos atendidos em dados originais (poucos casos de risco)
STATUS_EXP = "candidato_experimental"

# ======================================================================= limpeza de texto
# Stopwords PT embutidas (equivalentes à lista NLTK; evita download em runtime).
# Negações são preservadas (inclui "sem", que a Sprint 3 removia).
NEGACOES = frozenset({"não", "nem", "nunca", "jamais", "nada", "nenhum", "nenhuma", "sem"})
_STOP = """a à ao aos aquela aquelas aquele aqueles aquilo as às até com como da das de dela delas
dele deles depois do dos e é ela elas ele eles em entre era eram éramos essa essas esse esses esta
está estamos estão estar estas estava estavam estávamos este esteja estejam estejamos estes esteve
estive estivemos estiver estivera estiveram estivéramos estiverem estivermos estivesse estivessem
estivéssemos estou eu foi fomos for fora foram fôramos forem formos fosse fossem fôssemos fui há
haja hajam hajamos hão havemos haver hei houve houvemos houver houvera houverá houveram houvéramos
houverão houverei houverem houveremos houveria houveriam houveríamos houvermos houvesse houvessem
houvéssemos isso isto já lhe lhes mais mas me mesmo meu meus minha minhas muito na não nas nem no
nos nós nossa nossas nosso nossos num numa o os ou para pela pelas pelo pelos por qual quando que
quem são se seja sejam sejamos sem será serão serei seremos seria seriam seríamos seu seus só somos
sou sua suas também te tem tém tinha tinham tínhamos tive tivemos tiver tivera tiveram tivéramos
tiverem tivermos tivesse tivessem tivéssemos tu tua tuas um uma você vocês vos""".split()
STOPWORDS_PT = frozenset(_STOP) - NEGACOES

_RE_LOCUTOR = re.compile(r"\[LOCUTOR\s*\d+\]\s*:?", re.I)
_RE_PLACEHOLDER = re.compile(r"\[(PESSOA|EMPRESA|LOCAL)\]", re.I)
_RE_NAO_LETRA = re.compile(r"[^a-záàâãéêíóôõúüç\s]")


def remover_marcacoes(texto):
    """Remove [LOCUTOR N]: e placeholders anonimizados, mantendo quebras de linha."""
    texto = _RE_LOCUTOR.sub(" ", texto)
    return _RE_PLACEHOLDER.sub(" ", texto)


@functools.lru_cache(maxsize=4096)   # a mesma transcrição é limpa muitas vezes (folds, grades): cache só acelera
def _limpar_str(texto):
    texto = remover_marcacoes(texto).lower()
    texto = _RE_NAO_LETRA.sub(" ", texto)
    return " ".join(t for t in texto.split() if t not in STOPWORDS_PT and len(t) >= 3)


def limpar_texto(texto):
    """Limpeza textual única (treino, notebook e inferência). Texto inválido/vazio -> ''."""
    if not isinstance(texto, str) or not texto.strip():
        return ""
    return _limpar_str(texto)


def _parece_dado_estruturado(texto):
    """JSON/registros colados no lugar de uma transcrição (ex.: linhas do ANON_transcricao.json)."""
    t = texto.strip()
    if t[:1] in ("{", "["):
        try:
            return isinstance(json.loads(t), (dict, list))
        except ValueError:
            pass
    return bool(re.search(r'"(ID_MEETING|ANON_TRANSCRICAO|NOTA_NPS|DT_MEETING)"\s*:', t))


def validar_transcricao(texto):
    """Retorna (ok, motivo). Usada pelo scoring, pelo app e pelo CLI."""
    if not isinstance(texto, str):
        return False, "Entrada inválida: a transcrição deve ser um texto."
    if not texto.strip():
        return False, "A transcrição está vazia."
    if _parece_dado_estruturado(texto):
        return False, "A entrada parece dados estruturados (JSON/registros). Cole apenas o texto da transcrição."
    if len(texto.strip()) < MIN_CHARS:
        return False, f"Transcrição muito curta (mínimo de {MIN_CHARS} caracteres)."
    if len(texto) > MAX_CHARS:
        return False, f"Transcrição longa demais (máximo de {MAX_CHARS} caracteres)."
    if len(limpar_texto(texto).split()) < MIN_TOKENS:
        return False, "A transcrição não tem conteúdo textual suficiente após a limpeza."
    return True, ""


# ================================================================ base SINTÉTICA (500 linhas)
# Geração determinística (semente fixa). Cada transcrição é montada a partir de um cenário latente
# (satisfeito, problema técnico, dúvida comercial, ...) e de milhares de combinações de frases e
# espaços (produto, área, números, prazos). O NPS é sorteado DEPOIS do cenário, com sobreposição
# deliberada entre cenários (há clientes insatisfeitos que falam de modo neutro e vice-versa) —
# por isso o texto não determina o NPS. O NPS nunca aparece no texto e não é usado como preditor.
_PRODUTOS_SINT = {
    "Protheus": ["a folha de pagamento", "o ponto eletrônico", "a gestão de pessoas", "o controle de férias"],
    "RM": ["a folha de pagamento", "o ponto eletrônico", "o recrutamento", "a admissão de colaboradores"],
    "Fluig": ["o fluxo de aprovação", "a gestão de documentos", "a solicitação de férias",
              "o onboarding de colaboradores"],
}
_EXTRAS = ["recrutamento", "avaliação de desempenho", "gestão de benefícios", "portal do colaborador",
           "controle de treinamentos", "banco de horas"]
_AREAS_EQ = ["RH", "departamento pessoal", "TI", "financeiro", "contabilidade", "operações"]
_SEG_DESC = {"Varejo": "varejo", "Serviços": "prestação de serviços", "Indústria": "indústria",
             "Saúde": "saúde", "Educação": "educação"}
_SEG_PESOS = {"Varejo": 0.24, "Serviços": 0.26, "Indústria": 0.18, "Saúde": 0.17, "Educação": 0.15}
_FAIXAS = ["Até 100K", "100K-500K", "500K-1M", "1M-5M"]
_CNAE = {"Varejo": [4711302, 4713004, 4781400], "Serviços": [6201501, 7020400, 8211300],
         "Indústria": [1091102, 2229399, 2599399], "Saúde": [8610101, 8630504, 8650099],
         "Educação": [8513900, 8532500, 8599604]}
_UF = {"SP": 0.30, "MG": 0.12, "RJ": 0.10, "RS": 0.09, "PR": 0.09, "SC": 0.07, "BA": 0.06, "GO": 0.05,
       "PE": 0.04, "CE": 0.04, "DF": 0.04}
_CAPITAL = {"SP": "São Paulo", "MG": "Belo Horizonte", "RJ": "Rio de Janeiro", "RS": "Porto Alegre",
            "PR": "Curitiba", "SC": "Florianópolis", "BA": "Salvador", "GO": "Goiânia", "PE": "Recife",
            "CE": "Fortaleza", "DF": "Brasília"}

_ABERTURAS = [
    "Boa tarde, [PESSOA]! Obrigado por reservar esse horário. Hoje a ideia é falar sobre o {P} e como está a experiência aí na [EMPRESA].",
    "Bom dia, [PESSOA]. Tudo bem? Vamos começar falando do {P}?",
    "Olá, [PESSOA]! Antes de tudo, obrigado pelo tempo. Queria entender como o {P} tem atendido vocês.",
    "Boa noite, [PESSOA]. Obrigado por aceitar a reunião. Como estão as coisas por aí?",
    "Oi, [PESSOA], tudo certo? Vou direto ao ponto: quero ouvir como está a operação com o {P}.",
    "Bom dia! Obrigado por conectar. Pode me contar como anda {A} na [EMPRESA]?",
    "Boa tarde, [PESSOA]. Esta é a nossa conversa de acompanhamento sobre o {P}. Como têm sido as últimas semanas?",
]
_PERGUNTAS = [
    "Pode detalhar um pouco mais?",
    "Entendi. E como isso impacta o dia a dia do time?",
    "Certo, anotei aqui. Tem mais algum ponto que você queira destacar?",
    "Interessante. Isso acontece desde quando?",
    "E quem mais está envolvido nisso aí na [EMPRESA]?",
    "Perfeito, vamos por partes. O que mais?",
    "Faz sentido. Me conta um pouco mais do contexto.",
]
_RECONHECIMENTOS = [
    "Perfeito, anotei tudo por aqui.",
    "Certo, vou registrar isso e acompanhar de perto.",
    "Entendido. Vamos dar o encaminhamento interno.",
    "Anotado. Qualquer novidade eu te aviso.",
]
_DESPEDIDAS = [
    "Obrigado, [PESSOA]! Qualquer coisa me chame.",
    "Valeu pelo tempo. Envio um resumo por e-mail ainda hoje.",
    "Fechado então. Um abraço e até a próxima.",
    "Agradeço! Vou registrar tudo e a gente se fala em breve.",
]
_CONTEXTO = [
    "Somos uma empresa de {seg_desc}, com cerca de {n} colaboradores e {filiais} unidades, a maior em [LOCAL].",
    "Atuamos no segmento de {seg_desc} há mais de {anos} anos e hoje o time de {area_eq} cuida de tudo isso.",
]
_TRANSVERSAIS = [
    "Vou precisar alinhar isso com o [PESSOA] da diretoria antes de decidir.",
    "Hoje o time de {area_eq} é pequeno, são {rh} pessoas cuidando de tudo.",
    "Aqui na [EMPRESA] a gente atende clientes em várias cidades, inclusive em [LOCAL].",
    "Só lembrando que a nossa competência fecha no dia {dia}.",
    "O eSocial também nos preocupa, principalmente por causa dos prazos de envio.",
    "A gente usava planilhas antes, e ainda mantemos algumas só para conferência.",
    "O [PESSOA] do financeiro também participa das nossas conversas sobre o sistema.",
    "No segmento de {seg_desc} a rotina é bem corrida, então o tempo é curto.",
]

_COMENTARIOS = {   # frases curtas e opinativas, em vocabulário variado, sorteadas conforme o tom do cenário
    "pos": ["Excelente trabalho do time de vocês.", "Recomendo a solução para outras empresas do nosso grupo.",
            "Estamos muito satisfeitos com a parceria.", "O sistema é ótimo e o atendimento merece elogio.",
            "Foi uma ótima escolha, sem dúvida.", "Superou as nossas expectativas.",
            "Os resultados apareceram rápido e isso ajudou muito a gente."],
    "neg": ["Está péssimo, não dá para trabalhar assim.", "É uma situação frustrante para todo mundo aqui.",
            "Foi decepcionante, esperávamos muito mais.", "O sistema não funciona como deveria.",
            "Isso é inaceitável para uma empresa do tamanho de vocês.",
            "Estamos insatisfeitos e já reclamamos várias vezes.",
            "O problema se repete e ninguém resolve de vez.", "Essa demora toda está nos custando dinheiro."],
    "neu": ["Por ora é só isso.", "Sem pressa, mas queria registrar.", "Vamos ver como fica nas próximas semanas.",
            "Depois a gente conversa melhor sobre isso."],
    "res": ["No fim das contas foi bem resolvido.", "A equipe foi muito atenciosa até resolver.",
            "Agora tudo funciona normalmente."],
}
_TOM = {"sat_expansao": ["pos"], "sat_rotina": ["pos"], "duvida_comercial": ["neu"], "problema_andamento": ["neg"],
        "problema_resolvido": ["res", "pos"], "insat_suporte": ["neg"], "misto": ["pos", "neg"],
        "neutro": ["neu"], "implantacao": ["neu", "pos", "neg"], "treinamento": ["pos", "neu"]}

_CENARIOS = {
    "sat_expansao": dict(
        abre=["Está indo muito bem! O {P} resolveu boa parte dos nossos gargalos em {A}, a equipe está bem satisfeita.",
              "Estamos contentes com o resultado. Depois que implantamos o {P}, {A} ficou muito mais simples de tocar.",
              "Ótimo, de verdade. O {P} entrou na rotina e ninguém quer voltar para as planilhas.",
              "Muito bom o retorno aqui dentro. {A_cap} que levava dias agora roda em poucas horas.",
              "Tranquilo e com bons resultados. Inclusive quero conversar sobre ampliar o uso do {P}."],
        detalhes=["Estamos abrindo mais {filiais} unidades este ano e gostaria de levar o {P} para todas elas.",
                  "Podemos pensar em uma proposta para incluir mais {n} usuários no {P}?",
                  "Queria conhecer o módulo de {extra} que vocês comentaram, dá para agendar uma demonstração?",
                  "O pessoal do financeiro viu o resultado e também quer usar. Consegue me passar um orçamento?",
                  "Vamos contratar mais {n} colaboradores até o fim do ano, então preciso ampliar a licença.",
                  "Se o valor couber no orçamento do próximo trimestre, quero fechar a ampliação ainda este mês.",
                  "Estou pensando em levar também {A2} para dentro do {P}. Vocês atendem isso?",
                  "O retorno da equipe foi tão bom que a diretoria pediu para expandir para outras áreas."],
        consultor=["Que bom ouvir isso! Vou preparar uma proposta de ampliação e te envio até {prazo}.",
                   "Perfeito. Posso agendar uma demonstração do módulo com o time na semana que vem.",
                   "Ótima notícia. Já levanto o orçamento para as novas unidades e te mando por e-mail."],
        fecha=["Combinado, fico no aguardo da proposta. Obrigado pelo apoio de vocês.",
               "Perfeito, pode agendar. Já aviso o pessoal da diretoria.",
               "Ótimo! Estamos muito satisfeitos e pretendemos recomendar a solução para outros parceiros."],
        p_risco=0.04, nps_alto="sat"),
    "sat_rotina": dict(
        abre=["Tudo certo por aqui. O {P} está funcionando bem e {A} roda sem sobressaltos.",
              "Sem queixas. {A_cap} tem fechado no prazo e a equipe aprendeu rápido.",
              "Está estável, o pessoal gosta da ferramenta e o suporte sempre nos atendeu bem.",
              "Bem tranquilo. Tivemos o fechamento do mês sem nenhum erro, o que não acontecia antes."],
        detalhes=["A equipe de {area_eq} usa todos os dias e o retorno tem sido positivo.",
                  "O treinamento que vocês deram ajudou bastante, hoje todo mundo sabe usar sem ajuda.",
                  "Só uma sugestão: um relatório mais fácil de exportar, mas nada que atrapalhe.",
                  "As atualizações chegam sem problemas e o time de TI não precisa mais ficar de plantão.",
                  "O atendimento do suporte é rápido; quando abrimos chamado, resolvem no mesmo dia.",
                  "A gente percebeu um ganho de uns {horas} horas por semana na rotina de {A}.",
                  "O pessoal elogiou a facilidade de uso e a clareza dos relatórios."],
        consultor=["Fico feliz! Se aparecer qualquer dúvida, já sabe onde nos encontrar.",
                   "Ótimo saber. Vou registrar sua sugestão de relatório com o time de produto.",
                   "Obrigado pelo retorno. Seguimos acompanhando para garantir que continue assim."],
        fecha=["Combinado, a gente conversa de novo no próximo trimestre.",
               "Valeu, qualquer coisa eu aviso. Obrigado pela atenção.",
               "Perfeito, por enquanto está tudo certo. Agradeço o contato."],
        p_risco=0.06, nps_alto="sat"),
    "duvida_comercial": dict(
        abre=["Estamos avaliando o {P} e queria entender melhor como funcionam o plano e o preço.",
              "Queria saber quanto custa ampliar a licença do {P} e qual o prazo de implantação.",
              "Tenho algumas dúvidas sobre contrato, reajuste e o que está incluso no suporte.",
              "Fiquei com dúvida em relação à cobrança por usuário e ao desconto para pagamento anual.",
              "Preciso de uma visão de valores para levar à diretoria, ainda estamos em fase de comparação."],
        detalhes=["Hoje temos {n} funcionários e {filiais} unidades; como seria o enquadramento no plano?",
                  "O valor da implantação é cobrado à parte ou entra na mensalidade?",
                  "Qual o prazo médio para colocar {A} em produção?",
                  "Existe algum desconto se contratarmos mais de um módulo ao mesmo tempo?",
                  "Estamos olhando outras opções também, mas queremos entender bem a de vocês antes de decidir.",
                  "Preciso comparar o que está incluso no suporte com o que a gente tem hoje."],
        consultor=["Claro, vou detalhar. O plano é por faixa de usuários e a implantação costuma levar entre {dias} dias.",
                   "Posso te enviar uma tabela com as faixas e as condições de desconto ainda hoje.",
                   "Entendo. Vamos montar um comparativo para você levar à diretoria."],
        fecha=["Pode enviar, por favor. Depois marcamos outra conversa para fechar os detalhes.",
               "Ok, vou analisar com a diretoria e te retorno na próxima semana.",
               "Certo, aguardo os valores por e-mail. Obrigado."],
        p_risco=0.35, nps_alto="mild"),
    "problema_andamento": dict(
        abre=["Está complicado. {A_cap} no {P} vem dando erro quase todo dia e a equipe perde horas refazendo o trabalho.",
              "Estamos com um problema sério: o {P} fica lento no fechamento e às vezes trava no meio do processo.",
              "Não está bom. A integração do {P} com o financeiro falha e a gente precisa conferir tudo na mão.",
              "Para ser sincero, estamos bastante frustrados. {A_cap} apresenta instabilidade desde a última atualização.",
              "Tivemos falhas repetidas na marcação de ponto e os colaboradores estão reclamando."],
        detalhes=["Já abrimos chamado há {dias_c} dias e até agora não tivemos uma solução definitiva.",
                  "O erro aparece principalmente na virada do mês, justamente quando mais precisamos do sistema.",
                  "Meu time está refazendo manualmente o que o sistema deveria calcular, isso gera retrabalho e atraso.",
                  "A lentidão piora no fim da tarde, quando todo mundo acessa ao mesmo tempo.",
                  "Já reiniciamos o servidor, aplicamos o patch e o problema continua acontecendo.",
                  "Isso está impactando o pagamento dos colaboradores e a diretoria já cobrou explicação.",
                  "A sincronização com o relógio de ponto está falhando e perdemos marcações."],
        consultor=["Entendo a gravidade. Vou escalar o chamado para o time técnico e pedir prioridade.",
                   "Peço desculpas pelo transtorno. Vou acompanhar pessoalmente até a correção.",
                   "Vou abrir um incidente com o time de produto e te passo um retorno ainda hoje."],
        fecha=["Preciso de uma posição rápida, porque isso não pode continuar assim.",
               "Ok, fico aguardando o retorno. Se não melhorar, vamos ter que repensar o contrato.",
               "Certo, mas já adianto que estamos perdendo a paciência.",
               "Combinado, mas precisamos de um prazo concreto para a solução."],
        p_risco=0.85, nps_alto="mild", severo=True),
    "problema_resolvido": dict(
        abre=["Tivemos um problema com {A} no mês passado, mas o suporte resolveu e agora está funcionando.",
              "A instabilidade que tínhamos no {P} foi corrigida depois do patch, desde então não deu mais erro.",
              "Estávamos com dificuldade na integração, mas a equipe de vocês solucionou rápido."],
        detalhes=["O chamado foi encerrado em {dias_r} dias e o time técnico explicou a causa do erro.",
                  "Depois da correção, o fechamento rodou normalmente e ninguém mais reclamou.",
                  "No começo ficamos preocupados, mas o atendimento foi ágil e transparente.",
                  "Só pedimos que avisem antes das próximas atualizações para a gente se preparar.",
                  "Agora está estável; mantivemos o monitoramento por precaução.",
                  "A lentidão acabou depois que ajustaram o banco de dados."],
        consultor=["Fico contente que tenha sido resolvido. Vamos seguir monitorando.",
                   "Obrigado por avisar. Vou garantir que o aviso das próximas atualizações chegue com antecedência.",
                   "Ótimo. Se algo voltar a acontecer, abra o chamado e cite este caso."],
        fecha=["Está tudo certo agora, obrigado pela agilidade.",
               "Resolvido por ora; se voltar a acontecer, a gente conversa.",
               "Combinado. Obrigado pelo acompanhamento."],
        p_risco=0.30, nps_alto="mild"),
    "insat_suporte": dict(
        abre=["Sinceramente, estou muito insatisfeito com o atendimento. Abrimos chamados e ninguém responde.",
              "O suporte é muito demorado. Já foram {dias_c} dias sem retorno e isso é inaceitável.",
              "A gente esperava mais da TOTVS. O {P} não entregou o que foi prometido na venda.",
              "Estou decepcionado. Pagamos caro e o {P} continua com os mesmos problemas de sempre."],
        detalhes=["Já recebi proposta de um concorrente com preço menor e com atendimento mais próximo.",
                  "Se continuar assim, vamos avaliar o cancelamento do contrato no fim do ciclo.",
                  "Cada vez que ligo, falo com uma pessoa diferente e preciso explicar tudo de novo.",
                  "O treinamento prometido nunca aconteceu e meu time não sabe usar direito.",
                  "Perdemos o prazo de uma obrigação por causa de um erro que já tínhamos reportado.",
                  "A diretoria está pedindo para trocar de fornecedor, e eu estou sem argumento para defender vocês.",
                  "Isso gera prejuízo para a operação e não vejo ninguém assumindo a responsabilidade."],
        consultor=["Você tem toda razão em reclamar. Vou levar isso para a gerência hoje mesmo.",
                   "Sinto muito pela experiência. Quero entender o que aconteceu e montar um plano de recuperação.",
                   "Vou designar um responsável único para o seu caso."],
        fecha=["Ok, mas preciso ver resultado de verdade nos próximos dias, senão vamos seguir outro caminho.",
               "Espero que dessa vez seja diferente, porque minha paciência está no limite.",
               "Está bem, aguardo. Mas não tenho muita esperança."],
        p_risco=0.94, nps_alto="mild", severo=True),
    "misto": dict(
        abre=["O {P} é bom, mas {A} ainda tem alguns problemas e demora para fechar.",
              "Tem pontos positivos, o sistema é completo, porém o suporte às vezes demora a responder.",
              "Gostamos da ferramenta, mas a implantação atrasou e isso deixou o time ansioso.",
              "Mais ou menos. Algumas coisas funcionam bem, outras ainda dão trabalho."],
        detalhes=["A parte de {A} é excelente; já a integração com o financeiro dá erro de vez em quando.",
                  "O sistema é completo, mas a tela é complicada e o pessoal teve dificuldade de aprender.",
                  "O suporte resolve, mas leva uns {dias_c} dias, então a gente fica parado esperando.",
                  "Os relatórios são ótimos, só o tempo de resposta às vezes é lento.",
                  "Estamos satisfeitos com o produto, só precisamos que o atendimento seja mais ágil."],
        consultor=["Entendo. Vamos montar um plano de melhorias para os pontos que você citou.",
                   "Obrigado pela franqueza. Vou levar o ponto do atendimento para a liderança.",
                   "Faz sentido. Posso acompanhar de perto as próximas semanas."],
        fecha=["Vamos seguindo. Se melhorar o atendimento, ficamos bem mais confortáveis.",
               "Ok, fico no aguardo do plano de melhorias.",
               "Tudo bem, mas gostaria de um acompanhamento mais próximo daqui para frente."],
        p_risco=0.55, nps_alto="mild"),
    "neutro": dict(
        abre=["Tudo bem por aqui, estamos na correria do fechamento do trimestre.",
              "Estamos seguindo. Preciso só alinhar uns pontos de agenda e documentação.",
              "Pode ser rápido? Tenho outra reunião logo em seguida.",
              "Estamos organizando as férias do pessoal, então o time está um pouco disperso."],
        detalhes=["Preciso do contato do time responsável pelo {P} para a próxima etapa.",
                  "Vou verificar com o pessoal e te retorno por e-mail.",
                  "Podemos remarcar para a semana que vem? Hoje o time de {area_eq} está sobrecarregado.",
                  "Por enquanto sem novidades, vou conversar internamente.",
                  "Pode me mandar um resumo dessa conversa por e-mail?"],
        consultor=["Sem problemas, ajusto a agenda e te mando um novo convite.",
                   "Claro, envio o contato e o resumo ainda hoje.",
                   "Combinado, fico à disposição para o que precisar."],
        fecha=["Pode ser, me envie um horário por e-mail.",
               "Certo, falamos na semana que vem. Obrigado.",
               "Ok, combinado."],
        p_risco=0.45, nps_alto="mild"),
    "implantacao": dict(
        abre=["Estamos na fase de implantação do {P} e queria alinhar o cronograma.",
              "A migração dos dados de {A} está em andamento e tem algumas pendências.",
              "O projeto está seguindo, mas com algum atraso em relação ao que foi combinado."],
        detalhes=["A homologação de {A} está prevista para {mes}, mas ainda faltam testes de integração.",
                  "O time de TI recebeu o treinamento e tirou boa parte das dúvidas.",
                  "Alguns cadastros vieram com inconsistência da base antiga e precisamos corrigir.",
                  "O consultor que acompanha o projeto tem sido atencioso, só estamos preocupados com o prazo.",
                  "Precisamos entrar em produção até o fim do mês por causa da competência da folha."],
        consultor=["Vamos revisar o cronograma juntos e priorizar o que é crítico para a entrada em produção.",
                   "Entendi. Posso alocar mais um consultor para acelerar a homologação.",
                   "Certo, vou pedir ao time de dados um relatório com as inconsistências."],
        fecha=["Combinado, vamos seguir o cronograma revisado.",
               "Ok, confirmo a próxima reunião de acompanhamento.",
               "Fico mais tranquilo com esse plano, mas vamos acompanhar de perto."],
        p_risco=0.55, nps_alto="mild"),
    "treinamento": dict(
        abre=["Queria agendar o treinamento do time de {area_eq} no {P}.",
              "Fizemos o treinamento ontem e o pessoal gostou bastante da didática.",
              "Preciso de uma nova turma de treinamento para quem entrou este mês."],
        detalhes=["Quantas pessoas podem participar de cada turma?",
                  "O conteúdo cobriu bem {A}, mas faltou um pouco de prática com relatórios.",
                  "Seria bom ter material de apoio para quem entrar depois.",
                  "O instrutor tirou todas as dúvidas e o time já está usando no dia a dia.",
                  "Prefiro turmas online, porque as unidades ficam em cidades diferentes."],
        consultor=["Cada turma comporta até {n_turma} pessoas e podemos fazer online ou presencial.",
                   "Posso enviar o material de apoio e a gravação das sessões.",
                   "Vou reservar a agenda do instrutor para {prazo}."],
        fecha=["Perfeito, pode reservar a turma.",
               "Ótimo, aguardo o material por e-mail.",
               "Combinado, obrigado pela flexibilidade."],
        p_risco=0.25, nps_alto="mild"),
}
_PESOS_CENARIO = {"sat_expansao": 10, "sat_rotina": 12, "duvida_comercial": 10, "problema_andamento": 16,
                  "problema_resolvido": 10, "insat_suporte": 12, "misto": 10, "neutro": 8,
                  "implantacao": 7, "treinamento": 5}
_NOTAS_RISCO_SEVERO = [0.12, 0.08, 0.12, 0.14, 0.16, 0.18, 0.20]          # NPS 0..6
_NOTAS_RISCO_LEVE = [0.02, 0.02, 0.04, 0.08, 0.14, 0.30, 0.40]
_NOTAS_OK_ALTO = [0.10, 0.20, 0.30, 0.40]                                  # NPS 7..10
_NOTAS_OK_LEVE = [0.35, 0.35, 0.20, 0.10]


def _slots(rng, seg):
    prod = rng.choice(list(_PRODUTOS_SINT))
    area, area2 = rng.sample(_PRODUTOS_SINT[prod], 2)
    return {"P": prod, "A": area, "A_cap": area[0].upper() + area[1:], "A2": area2,
            "area_eq": rng.choice(_AREAS_EQ), "n": rng.choice([35, 60, 80, 120, 180, 250, 400, 650, 900]),
            "filiais": rng.randint(2, 12), "dias": rng.choice(["30 a 45", "45 a 60", "60 a 90"]),
            "dias_c": rng.randint(5, 25), "dias_r": rng.randint(1, 4), "horas": rng.randint(4, 20),
            "extra": rng.choice(_EXTRAS), "mes": rng.choice(["novembro", "dezembro", "janeiro", "março"]),
            "prazo": rng.choice(["amanhã", "sexta-feira", "o fim da semana", "quarta-feira", "terça-feira"]),
            "anos": rng.randint(3, 30), "rh": rng.randint(2, 9), "dia": rng.randint(20, 30),
            "n_turma": rng.choice([8, 10, 12, 15]), "seg_desc": _SEG_DESC[seg]}


def _montar_transcricao(rng, cenario, seg):
    cfg, s = _CENARIOS[cenario], _slots(rng, seg)
    f = lambda t: t.format(**s)
    turnos = [(1, f(rng.choice(_ABERTURAS))), (2, f(rng.choice(cfg["abre"])))]
    consultor = rng.sample(cfg["consultor"], rng.randint(1, len(cfg["consultor"])))
    perguntas = rng.sample(_PERGUNTAS, len(_PERGUNTAS))   # embaralhadas; no máximo 5 são usadas
    detalhes = rng.sample(cfg["detalhes"], rng.randint(2, min(4, len(cfg["detalhes"]))))
    for i, det in enumerate(detalhes):
        turnos.append((1, f(consultor.pop(0)) if consultor and i % 2 == 1 else perguntas.pop(0)))
        turnos.append((2, f(det)))
        if i == 0 and rng.random() < 0.55:
            turnos.append((1, perguntas.pop(0)))
            turnos.append((2, f(rng.choice(_CONTEXTO))))
        if rng.random() < 0.30:
            turnos.append((2, f(rng.choice(_TRANSVERSAIS))))
    if rng.random() < 0.65:   # comentário opinativo logo após uma fala do cliente
        tom = rng.choice(_TOM[cenario])
        i = rng.choice([k for k, (l, _) in enumerate(turnos) if l == 2 and k >= 1])
        turnos.insert(i + 1, (2, rng.choice(_COMENTARIOS[tom])))
    turnos.append((1, f(consultor.pop(0)) if consultor else rng.choice(_RECONHECIMENTOS)))
    turnos.append((2, f(rng.choice(cfg["fecha"]))))
    turnos.append((1, rng.choice(_DESPEDIDAS)))
    if rng.random() < 0.25:  # fala coloquial de transcrição automática
        turnos = [(l, t.replace(" está ", " tá ").replace(" para ", " pra ")) if l == 2 else (l, t) for l, t in turnos]
    usadas, unicos = set(), []
    for l, t in turnos:          # nenhuma fala idêntica repetida dentro da mesma transcrição
        if t not in usadas:
            usadas.add(t)
            unicos.append((l, t))
    turnos = unicos
    texto = "\n".join(f"[LOCUTOR {l}]: {t}" for l, t in turnos)
    assert "{" not in texto and "}" not in texto, f"espaço de template não preenchido: {texto}"
    return texto, len(turnos)


def _sortear_nps(rng, cfg, risco):
    if risco:
        pesos = _NOTAS_RISCO_SEVERO if cfg.get("severo") else _NOTAS_RISCO_LEVE
        return rng.choices(range(0, 7), weights=pesos)[0]
    return rng.choices(range(7, 11), weights=_NOTAS_OK_ALTO if cfg["nps_alto"] == "sat" else _NOTAS_OK_LEVE)[0]


def gerar_base_sintetica(n=500, seed=RANDOM_STATE):
    """Gera `n` reuniões FICTÍCIAS com as 18 colunas do CSV original e transcrições todas distintas."""
    rng = random.Random(seed)
    nomes, pesos = list(_PESOS_CENARIO), list(_PESOS_CENARIO.values())
    segs, segp = list(_SEG_PESOS), list(_SEG_PESOS.values())
    ufs, ufp = list(_UF), list(_UF.values())
    inicio = datetime(2025, 1, 6)
    vistos, linhas = set(), []
    while len(linhas) < n:
        cenario = rng.choices(nomes, weights=pesos)[0]
        seg = rng.choices(segs, weights=segp)[0]
        texto, n_turnos = _montar_transcricao(rng, cenario, seg)
        if texto in vistos:           # sem textos repetidos
            continue
        vistos.add(texto)
        cfg = _CENARIOS[cenario]
        nps = _sortear_nps(rng, cfg, rng.random() < cfg["p_risco"])
        dt = inicio + timedelta(days=rng.randint(0, 270))
        completo = rng.random() < 0.85
        uf = rng.choices(ufs, weights=ufp)[0]
        dur = max(8 * 60, int(n_turnos * rng.uniform(150, 420)))
        linhas.append({
            "ID_MEETING": 1000000 + len(linhas),
            "DT_MEETING": dt.strftime("%Y-%m-%d 00:00:00"),
            "FORMATO_MEETING": rng.choice(["VIDEO", "VIDEO", "ÁUDIO"]),
            "ID_STATUS_MEETING": 3 if completo else 1,
            "STATUS_MEETING": "COMPLETED" if completo else "IN_PROGRESS",
            "DURACAO_MEETING": f"{dur // 3600:02d}:{dur % 3600 // 60:02d}:{dur % 60:02d}",
            "CODT": f"T{rng.randint(10000, 99999)}",
            "TP_RECURSO": rng.choice(["OPERACIONAL", "VENDAS", "SUCESSO_CLIENTE", "SUPORTE"]),
            "FLG_EXTERNO": rng.random() < 0.7,
            "DT_CRIACAO": (dt + timedelta(hours=rng.randint(1, 20))).strftime("%Y-%m-%d %H:%M:%S"),
            COLUNA_TEXTO: texto,
            "UF": uf,
            "CNAE": rng.choice(_CNAE[seg]),
            "NOME_UNIDADE": f"Unidade {_CAPITAL[uf]}",
            "NOME_SEGMENTO": seg,
            "FAIXA_FATURAMENTO_CLIENTE": rng.choice(_FAIXAS),
            "DT_ULTIMA_PESQUISA": (dt + timedelta(days=rng.randint(1, 30))).strftime("%Y-%m-%d 00:00:00"),
            COLUNA_NPS: nps,
        })
    return pd.DataFrame(linhas)


def salvar_base_sintetica(caminho=CSV_PADRAO, n=500, seed=RANDOM_STATE):
    df = gerar_base_sintetica(n, seed)
    df.to_csv(caminho, index=False, encoding="utf-8")
    return df


# ================================================================================== dados
def carregar_json(caminho=JSON_PADRAO):
    """Lê o JSON original LINHA A LINHA (JSON Lines; aceita também uma lista JSON) e normaliza as colunas:
    FAIXA_FATURAMENTO_CLIENTE_EC -> FAIXA_FATURAMENTO_CLIENTE; NOTA_NPS vira numérico (vazio/inválido -> NaN).
    Linhas ilegíveis são contadas em df.attrs['linhas_ilegiveis'] e descartadas."""
    texto = Path(caminho).read_text(encoding="utf-8")
    registros, ilegiveis = [], 0
    if texto.lstrip().startswith("["):
        registros = json.loads(texto)
    else:
        for linha in texto.splitlines():
            if linha.strip():
                try:
                    registros.append(json.loads(linha))
                except json.JSONDecodeError:
                    ilegiveis += 1
    df = pd.DataFrame(registros).rename(columns={"FAIXA_FATURAMENTO_CLIENTE_EC": "FAIXA_FATURAMENTO_CLIENTE"})
    df[COLUNA_NPS] = pd.to_numeric(df.get(COLUNA_NPS), errors="coerce")
    df.attrs["linhas_ilegiveis"] = ilegiveis
    return df


def _grupos_sem_vazamento(df):
    """Id de grupo por linha: reuniões do MESMO cliente (CODT) ou com texto idêntico ficam juntas
    (componentes conexas via union-find), para que nada do mesmo cliente/texto caia em treino e teste."""
    pai = {}

    def achar(x):
        pai.setdefault(x, x)
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    for cod, txt in zip(df["CODT"] if "CODT" in df else [None] * len(df), df["TEXTO_LIMPO"]):
        a = achar(("t", txt))
        if isinstance(cod, str) and cod.strip():
            pai[achar(("c", cod))] = a
    return pd.Series([achar(("t", t)) for t in df["TEXTO_LIMPO"]]).factorize()[0]


def validar_base(df):
    """Valida o DataFrame (JSON ou CSV) e devolve (df_limpo, relatorio). Registros SEM NPS válido (0-10)
    são excluídos do treino supervisionado: nenhum rótulo é inventado para eles."""
    faltando = {COLUNA_TEXTO, COLUNA_NPS} - set(df.columns)
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(faltando)}")
    rel = {"linhas_originais": len(df), "linhas_ilegiveis": int(df.attrs.get("linhas_ilegiveis", 0))}
    d = df.copy()
    d[COLUNA_NPS] = pd.to_numeric(d[COLUNA_NPS], errors="coerce")
    rel["nps_invalido_ou_nulo"] = int((~d[COLUNA_NPS].between(0, 10)).sum())
    d = d[d[COLUNA_NPS].between(0, 10)].copy()
    n0 = len(d)
    if "ID_MEETING" in d:   # a mesma reunião repetida infla amostra e métricas
        d = d.drop_duplicates(subset="ID_MEETING", keep="first")
    rel["registros_id_meeting_repetido_removidos"] = int(n0 - len(d))
    d["TEXTO_LIMPO"] = d[COLUNA_TEXTO].apply(limpar_texto)
    vazios = d["TEXTO_LIMPO"].str.split().str.len().fillna(0) < MIN_TOKENS
    rel["transcricoes_vazias_ou_curtas"] = int(vazios.sum())
    d = d[~vazios].copy()
    d["ALVO_RISCO"] = (d[COLUNA_NPS] < LIMIAR_NPS).astype(int)
    g = d.groupby("TEXTO_LIMPO")["ALVO_RISCO"]
    rel["linhas_validas"] = len(d)
    rel["risco"], rel["nao_risco"] = int(d["ALVO_RISCO"].sum()), int((1 - d["ALVO_RISCO"]).sum())
    rel["proporcao_risco"] = float(d["ALVO_RISCO"].mean())
    rel["textos_unicos"] = int(g.ngroups)
    rel["linhas_em_textos_duplicados"] = int(d.duplicated("TEXTO_LIMPO", keep=False).sum())
    rel["textos_com_rotulos_conflitantes"] = int((g.nunique() > 1).sum())
    d = d.reset_index(drop=True)
    d["GRUPO"] = _grupos_sem_vazamento(d)
    rel["grupos_cliente_ou_texto"] = int(d["GRUPO"].nunique())
    if "CODT" in d:
        rel["clientes_distintos"] = int(d["CODT"].nunique())
        rel["clientes_com_rotulos_conflitantes"] = int((d.groupby("CODT")["ALVO_RISCO"].nunique() > 1).sum())
    return d, rel


def base_padrao():
    """Preferência: dados ORIGINAIS rotulados (JSON). Sem o JSON, cai no CSV (que NÃO serve de base principal)."""
    return JSON_PADRAO if JSON_PADRAO.exists() else CSV_PADRAO


def carregar_base(caminho=None):
    """Carrega e valida a base. .json -> dados originais (fonte 'json_original'); .csv -> fonte 'csv'
    (mockado/sintético: apenas demonstração). O relatório traz a chave 'fonte'."""
    caminho = Path(caminho) if caminho else base_padrao()
    if caminho.suffix.lower() == ".json":
        df, fonte = carregar_json(caminho), "json_original"
    else:
        df, fonte = pd.read_csv(caminho), "csv"
    d, rel = validar_base(df)
    rel["fonte"], rel["arquivo"] = fonte, caminho.name
    return d, rel


def dividir_base(df, seed=RANDOM_STATE):
    """Split estratificado treino/teste (75/25) com semente fixa, em GRUPOS (mesmo cliente ou texto idêntico
    ficam do mesmo lado: StratifiedGroupKFold). Com poucos grupos (< MIN_GRUPOS_UNICOS) o agrupamento degenera:
    usa-se split estratificado simples e emite-se aviso (métricas tendem a ser otimistas).
    Retorna X_tr, X_te, y_tr, y_te, grupos_tr (ids de grupo para a validação cruzada)."""
    y = df["ALVO_RISCO"]
    grupos = df["GRUPO"].to_numpy() if "GRUPO" in df else df["TEXTO_LIMPO"].factorize()[0]
    n_grupos = len(np.unique(grupos))
    if n_grupos >= MIN_GRUPOS_UNICOS:
        cv = StratifiedGroupKFold(n_splits=TEST_SIZE_FOLDS, shuffle=True, random_state=seed)
        idx_tr, idx_te = next(cv.split(df, y, grupos))
    else:
        warnings.warn(f"Apenas {n_grupos} grupos (< {MIN_GRUPOS_UNICOS}): split estratificado "
                      "simples; duplicatas podem vazar entre treino e teste (métricas otimistas).")
        cv = StratifiedKFold(n_splits=TEST_SIZE_FOLDS, shuffle=True, random_state=seed)
        idx_tr, idx_te = next(cv.split(df, y))
        grupos = np.arange(len(df))
    tr, te = df.iloc[idx_tr], df.iloc[idx_te]
    return tr[COLUNA_TEXTO], te[COLUNA_TEXTO], tr["ALVO_RISCO"], te["ALVO_RISCO"], grupos[idx_tr]


# ===================================================================== modelos e avaliação
class LinearSVCScore(LinearSVC):
    """LinearSVC com predict_proba = sigmoide da margem. NÃO é probabilidade calibrada: serve apenas para
    ordenar (ROC/PR AUC, limiar escolhido por validação cruzada e score em percentil)."""

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1.0 - p, p])


_TFIDF_BASE = dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.90, max_features=5000)
# Variações de TF-IDF avaliadas SOMENTE por validação cruzada no desenvolvimento (word 1-2g; word 1g mais podado;
# word 1-2g mais podado; n-gramas de caracteres).
_TFIDF_CFGS = [
    _TFIDF_BASE,
    dict(analyzer="word", ngram_range=(1, 1), min_df=3, max_df=0.50, max_features=5000),
    dict(analyzer="word", ngram_range=(1, 2), min_df=5, max_df=0.50, max_features=3000),
    dict(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_df=0.90, max_features=20000),
]


def construir_pipeline(clf, memory=None):
    """Limpeza -> TF-IDF -> classificador. O vetorizador é ajustado só no treino (dentro de cada fold)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=limpar_texto, lowercase=False, sublinear_tf=True, **_TFIDF_BASE)),
        ("clf", clf),
    ], memory=memory)


def _grade(clf_grid):
    """Produto (variação de TF-IDF) x (hiperparâmetros do classificador), como lista de grades de um ponto."""
    return [{**{f"tfidf__{k}": [v] for k, v in cfg.items()}, **{k: [v] for k, v in combo.items()}}
            for cfg in _TFIDF_CFGS for combo in ParameterGrid(clf_grid)]


def candidatos(memory=None):
    """nome -> (pipeline, grade). O baseline não tem grade. `memory` guarda TF-IDF já ajustado por fold."""
    cp = lambda clf: construir_pipeline(clf, memory)
    return {
        "Baseline (Dummy)": (cp(DummyClassifier(strategy="prior")), None),
        "Regressao Logistica": (cp(LogisticRegression(class_weight="balanced", max_iter=2000,
                                                      random_state=RANDOM_STATE)),
                                _grade({"clf__C": [0.01, 0.1, 1.0]})),
        "Linear SVM": (cp(LinearSVCScore(class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000)),
                       _grade({"clf__C": [0.001, 0.01, 0.1]})),
        "Complement NB": (cp(ComplementNB()), _grade({"clf__alpha": [0.1, 0.5, 1.0]})),
        "Random Forest": (cp(RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                    random_state=RANDOM_STATE, n_jobs=-1)),
                          _grade({"clf__min_samples_leaf": [2, 5], "clf__max_depth": [None, 10]})),
    }


def metricas(y, proba, limiar):
    """Matriz de confusão + accuracy, precision, recall, F1, balanced accuracy, ROC AUC, PR AUC."""
    pred = (np.asarray(proba) >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "Accuracy": accuracy_score(y, pred), "Balanced Acc": balanced_accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "ROC AUC": roc_auc_score(y, proba), "PR AUC": average_precision_score(y, proba),
            "% previsto risco": float(pred.mean())}


def escolher_limiar(y, proba, espec_min=ESPECIFICIDADE_MIN):
    """Política de negócio: maior recall mantendo especificidade >= espec_min, calculada em previsões
    fora-da-amostra do TREINO (nunca no teste). Impede a solução trivial 'tudo é risco'."""
    fpr, tpr, thr = roc_curve(y, proba)
    ok = np.where(fpr <= 1 - espec_min)[0]
    i = ok[np.argmax(tpr[ok])]
    return float(min(thr[i], 1.0)) if np.isfinite(thr[i]) else 1.0


def _oof(modelo, X, y, grupos, seed):
    """Previsões fora-da-amostra por validação cruzada agrupada (o TF-IDF é refeito em cada fold)."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof, aucs = np.zeros(len(y)), []
    for a, b in cv.split(X, y, grupos):
        m = clone(modelo).fit(X.iloc[a], y.iloc[a])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
        if y.iloc[b].nunique() == 2:
            aucs.append(roc_auc_score(y.iloc[b], oof[b]))
    return oof, np.array(aucs)


def _ic_auc_bootstrap(y, proba, n=1000):
    rng, v, y = np.random.RandomState(RANDOM_STATE), [], np.asarray(y)
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(set(y[i])) == 2:
            v.append(roc_auc_score(y[i], np.asarray(proba)[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def avaliar_candidatos(X_tr, y_tr, g_tr, X_te, y_te):
    """Hiperparâmetros (inclui a variação de TF-IDF) e CV só no treino; o teste é usado uma única vez, para reportar."""
    linhas, modelos, probas_te, oof_tr = [], {}, {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for nome, (pipe, grade) in candidatos(joblib.Memory(tmp, verbose=0)).items():
            if grade:
                cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                gs = GridSearchCV(pipe, grade, scoring="roc_auc", cv=cv, n_jobs=1)
                gs.fit(X_tr, y_tr, groups=g_tr)
                pipe, params = gs.best_estimator_, gs.best_params_
            else:
                params = {}
            oof, aucs = _oof(pipe, X_tr, y_tr, g_tr, RANDOM_STATE + 1)
            limiar = 0.5 if nome.startswith("Baseline") else escolher_limiar(y_tr, oof)
            final = clone(pipe).fit(X_tr, y_tr)
            p_te = final.predict_proba(X_te)[:, 1]
            final.set_params(memory=None)   # artefato sem dependência do diretório temporário
            cvm = metricas(y_tr, oof, limiar)
            linhas.append({"Modelo": nome, "Hiperparametros": params, "Limiar": limiar,
                           "CV AUC media": float(aucs.mean()), "CV AUC desvio": float(aucs.std()),
                           "CV PR AUC": cvm["PR AUC"], "Prevalencia treino": float(y_tr.mean()),
                           "CV Balanced Acc": cvm["Balanced Acc"], "CV Recall": cvm["Recall"],
                           "CV % previsto risco": cvm["% previsto risco"],
                           "Teste": metricas(y_te, p_te, limiar), "Teste @0.5": metricas(y_te, p_te, 0.5),
                           "Teste AUC IC95": _ic_auc_bootstrap(y_te, p_te)})
            modelos[nome], probas_te[nome], oof_tr[nome] = final, p_te, oof
    return {"linhas": linhas, "modelos": modelos, "probas_teste": probas_te, "oof_treino": oof_tr}


def controle_rotulos_embaralhados(X_tr, y_tr, X_te, y_te, seed=RANDOM_STATE):
    """Controle negativo: treina a mesma Regressão Logística com rótulos EMBARALHADOS.
    Sem relação texto-rótulo o AUC deve ficar perto de 0,5 — se não ficar, há vazamento. Também mostra o
    comportamento da configuração da Sprint 3 (sem class_weight, limiar 0,5): sem sinal, ela tende a prever
    só a classe majoritária."""
    y_emb = pd.Series(np.random.RandomState(seed).permutation(y_tr.values), index=y_tr.index)
    saida = {}
    for nome, cw in (("LR Sprint 3 (sem class_weight)", None), ("LR Sprint 4 (balanced)", "balanced")):
        m = construir_pipeline(LogisticRegression(class_weight=cw, max_iter=2000, random_state=RANDOM_STATE))
        p_real = clone(m).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        p_emb = clone(m).fit(X_tr, y_emb).predict_proba(X_te)[:, 1]
        saida[nome] = {"rotulos reais: AUC": roc_auc_score(y_te, p_real),
                       "rotulos reais: % previsto risco @0.5": float((p_real >= 0.5).mean()),
                       "rotulos embaralhados: AUC": roc_auc_score(y_te, p_emb),
                       "rotulos embaralhados: % previsto risco @0.5": float((p_emb >= 0.5).mean())}
    return pd.DataFrame(saida).T


def controle_selecao_embaralhada(X_tr, y_tr, g_tr, n=3, seed=RANDOM_STATE):
    """Quanto a SELEÇÃO de hiperparâmetros/TF-IDF infla o AUC de validação cruzada quando NÃO há sinal?
    Embaralha o rótulo ENTRE CLIENTES/GRUPOS (o rótulo é constante por cliente), repete a mesma busca da
    Regressão Logística (variações de TF-IDF x C) e devolve o melhor AUC de CV de cada embaralhamento."""
    grupos = np.asarray(g_tr)
    por_grupo = pd.Series(np.asarray(y_tr)).groupby(grupos).max()
    saida = []
    with tempfile.TemporaryDirectory() as tmp:
        pipe, grade = candidatos(joblib.Memory(tmp, verbose=0))["Regressao Logistica"]
        for k in range(n):
            novo = pd.Series(np.random.RandomState(seed + k).permutation(por_grupo.values), index=por_grupo.index)
            y_emb = pd.Series(novo.loc[grupos].values, index=y_tr.index)
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            gs = GridSearchCV(pipe, grade, scoring="roc_auc", cv=cv, n_jobs=1).fit(X_tr, y_emb, groups=grupos)
            saida.append(float(gs.best_score_))
    return saida


def curva_politica(y_tr, oof, espec_grid=(0.5, 0.6, 0.7, 0.8)):
    """Trade-off da política de limiar usando SÓ previsões fora-da-amostra do treino: quanto de recall se
    perde ao exigir mais especificidade (menos alarmes falsos)."""
    linhas = []
    for e in espec_grid:
        lim = escolher_limiar(y_tr, oof, e)
        m = metricas(y_tr, oof, lim)
        linhas.append({"especificidade minima": e, "limiar": round(lim, 4), "Recall": m["Recall"],
                       "Precision": m["Precision"], "Balanced Acc": m["Balanced Acc"],
                       "% previsto risco": m["% previsto risco"]})
    return pd.DataFrame(linhas).round(3).set_index("especificidade minima")


def tabela_comparativa(res, chave="Teste"):
    cols = ["TN", "FP", "FN", "TP", "Accuracy", "Balanced Acc", "Precision", "Recall", "F1",
            "ROC AUC", "PR AUC", "% previsto risco"]
    out = pd.DataFrame([{"Modelo": r["Modelo"], "Limiar": r["Limiar"] if chave == "Teste" else 0.5,
                         **{c: r[chave][c] for c in cols}} for r in res["linhas"]])
    return out.set_index("Modelo").round(4)


def tabela_cv(res):
    cols = ["Limiar", "CV AUC media", "CV AUC desvio", "CV PR AUC", "Prevalencia treino", "CV Balanced Acc", "CV Recall", "CV % previsto risco"]
    return pd.DataFrame([{"Modelo": r["Modelo"], **{c: r[c] for c in cols}} for r in res["linhas"]]
                        ).set_index("Modelo").round(4)


ORDEM_SIMPLICIDADE = ["Regressao Logistica", "Linear SVM", "Complement NB", "Random Forest"]   # do mais simples/interpretável ao mais complexo


def selecionar_modelo(res, base_sintetica=False):
    """Seleção só com a validação cruzada do treino; o teste apenas classifica o status final.

    1. Elegível: AUC de CV >= 0,60, (média - desvio) > 0,5, PR AUC de CV > prevalência e % previsto como risco <= 95% (nada de 'tudo é
       risco').
    2. Entre os elegíveis, vale a regra do 1 desvio-padrão: modelos cuja AUC de CV está a menos de 1 desvio
       do melhor são considerados EMPATADOS e escolhe-se o mais simples/interpretável (Regressão Logística
       antes de Random Forest). A prioridade de negócio por recall é aplicada pelo LIMIAR (maior recall com
       especificidade >= 50%), não pela escolha do algoritmo: o recall num único limiar oscila entre folds.
    3. Se ninguém for elegível, escolhe-se o de maior AUC de CV, marcado como experimental.
    Com base sintética o status é sempre 'demonstração técnica'; com dados originais, no máximo 'experimental'."""
    base = next(r for r in res["linhas"] if r["Modelo"].startswith("Baseline"))
    reais = [r for r in res["linhas"] if not r["Modelo"].startswith("Baseline")]
    elegiveis = [r for r in reais if r["CV AUC media"] - r["CV AUC desvio"] > 0.5
                 and r["CV AUC media"] >= AUC_CV_MIN_ELEGIVEL and r["CV % previsto risco"] <= PREVISTO_RISCO_MAX
                 and r["CV PR AUC"] > r["Prevalencia treino"]]
    if elegiveis:
        melhor = max(elegiveis, key=lambda r: r["CV AUC media"])
        empatados = [r for r in elegiveis if r["CV AUC media"] >= melhor["CV AUC media"] - melhor["CV AUC desvio"]]
        escolhido = min(empatados, key=lambda r: ORDEM_SIMPLICIDADE.index(r["Modelo"]))
    else:
        escolhido = max(reais, key=lambda r: r["CV AUC media"])
    t, lo = escolhido["Teste"], escolhido["Teste AUC IC95"][0]
    supera = bool(t["Balanced Acc"] > base["Teste"]["Balanced Acc"] and lo > 0.5)
    criterios = bool(elegiveis) and supera and t["ROC AUC"] >= AUC_MIN_SATISFATORIO \
        and t["Balanced Acc"] >= BAL_ACC_MIN_SATISFATORIO
    if base_sintetica:
        status = STATUS_DEMO if criterios else STATUS_EXP
    else:   # dados originais, mas poucos casos de risco: no máximo "experimental" (nunca piloto/produção)
        status = STATUS_REAL if criterios else STATUS_EXP
    return {"modelo": escolhido["Modelo"], "status": status, "elegivel_cv": bool(elegiveis),
            "empatados": [r["Modelo"] for r in (empatados if elegiveis else [])],
            "supera_baseline_teste": supera, "criterios_minimos_atendidos": criterios,
            "base_sintetica": bool(base_sintetica), "limiar": escolhido["Limiar"]}


def listar_erros(df_ml, y_te, proba, limiar):
    """Falsos negativos e falsos positivos do hold-out (com a probabilidade atribuída)."""
    t = df_ml.loc[y_te.index].assign(PROBA=np.asarray(proba), PRED=(np.asarray(proba) >= limiar).astype(int))
    return (t[(t.ALVO_RISCO == 1) & (t.PRED == 0)].sort_values("PROBA"),
            t[(t.ALVO_RISCO == 0) & (t.PRED == 1)].sort_values("PROBA", ascending=False))


# ================================================================================ artefato
def salvar_artefato(res, selecao, rel_base, caminho=ARTEFATO_PADRAO, X_tr=None):
    r = next(x for x in res["linhas"] if x["Modelo"] == selecao["modelo"])
    pipe = res["modelos"][selecao["modelo"]]
    cobertura = {}
    if X_tr is not None:   # termos TF-IDF por transcrição de treino: referência para detectar texto curto demais
        nnz = np.diff(pipe.named_steps["tfidf"].transform(X_tr).indptr)
        p05 = float(np.percentile(nnz, 5))
        cobertura = {"nnz_mediana_treino": float(np.median(nnz)), "nnz_p05_treino": p05,
                     "nnz_minimo": max(10, int(round(COBERTURA_MIN_FRACAO * float(np.median(nnz)))))}
    art = {"pipeline": pipe, "modelo": selecao["modelo"], "limiar": selecao["limiar"],
           "status": selecao["status"], "base_sintetica": selecao["base_sintetica"],
           "criterios_minimos_atendidos": selecao["criterios_minimos_atendidos"], "calibrado": False,
           "supera_baseline_teste": selecao["supera_baseline_teste"],
           "politica_limiar": f"maior recall com especificidade >= {ESPECIFICIDADE_MIN:.0%} (OOF no treino)",
           "metricas_teste": r["Teste"], "auc_teste_ic95": r["Teste AUC IC95"],
           "cv_auc_media": r["CV AUC media"], "relatorio_base": rel_base,
           "oof_referencia": np.sort(res["oof_treino"][selecao["modelo"]]).tolist(),
           "vocabulario": int(len(pipe.named_steps["tfidf"].vocabulary_)), **cobertura,
           "treinado_em": datetime.now().isoformat(timespec="seconds"),
           "versoes": {"python": sys.version.split()[0], "sklearn": sklearn.__version__}}
    joblib.dump(art, caminho)
    return Path(caminho)


def carregar_artefato(caminho=ARTEFATO_PADRAO):
    """Carrega o artefato do projeto (joblib = pickle: só carregue arquivos gerados por você)."""
    caminho = Path(caminho)
    if not caminho.exists():
        return None
    art = joblib.load(caminho)
    if art.get("versoes", {}).get("sklearn") != sklearn.__version__:
        warnings.warn(f"Artefato treinado com scikit-learn {art['versoes'].get('sklearn')}; "
                      f"versão atual {sklearn.__version__}. Retreine com `python pipeline.py treinar`.")
    return art


def treinar_base(caminho=None, caminho_artefato=ARTEFATO_PADRAO, base_sintetica=None):
    """Treina, compara e salva o artefato. `base_sintetica=None`: True para CSV, False para o JSON original."""
    df, rel = carregar_base(caminho)
    if base_sintetica is None:
        base_sintetica = rel["fonte"] != "json_original"
    X_tr, X_te, y_tr, y_te, g_tr = dividir_base(df)
    res = avaliar_candidatos(X_tr, y_tr, g_tr, X_te, y_te)
    sel = selecionar_modelo(res, base_sintetica)
    salvar_artefato(res, sel, rel, caminho_artefato, X_tr=X_tr)
    if rel["fonte"] == "json_original":        # resultados reais para o app (não sobrescreve com a base de demonstração)
        exportar_resultados(df, rel, res, y_te, df_bruto=carregar_json())
    return res, sel, rel


def estabilidade_holdout(res, df, n=20):
    """Robustez da estimativa: repete a divisão agrupada com n sementes, mantendo os hiperparâmetros já escolhidos
    (só o treino original os escolheu; leve otimismo residual). Não altera modelo, limiar nem artefato."""
    linhas = []
    for seed in range(n):
        X_tr, X_te, y_tr, y_te, _ = dividir_base(df, seed=1000 + seed)
        for nome, mod in res["modelos"].items():
            p = clone(mod).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
            linhas.append({"Modelo": nome, "ROC AUC": roc_auc_score(y_te, p),
                           "PR AUC": average_precision_score(y_te, p), "Prevalência teste": float(y_te.mean())})
    return pd.DataFrame(linhas).groupby("Modelo", sort=False).agg(
        auc_media=("ROC AUC", "mean"), auc_desvio=("ROC AUC", "std"), auc_min=("ROC AUC", "min"),
        auc_max=("ROC AUC", "max"), pr_auc_media=("PR AUC", "mean"), prevalencia_media=("Prevalência teste", "mean"))


def avaliar_artefato(caminho=None, caminho_artefato=ARTEFATO_PADRAO):
    """Reaplica o artefato salvo ao MESMO hold-out (a divisão é determinística) e confere se as métricas
    batem com as gravadas no treino. Detecta artefato desatualizado em relação à base."""
    art = carregar_artefato(caminho_artefato)
    if art is None:
        raise FileNotFoundError(f"Artefato não encontrado: {caminho_artefato}. Rode `python pipeline.py treinar`.")
    df, _ = carregar_base(caminho)
    _, X_te, _, y_te, _ = dividir_base(df)
    m = metricas(y_te, art["pipeline"].predict_proba(X_te)[:, 1], art["limiar"])
    iguais = all(abs(m[k] - art["metricas_teste"][k]) < 1e-9 for k in ("TN", "FP", "FN", "TP", "ROC AUC"))
    return {"metricas": m, "consistente_com_treino": bool(iguais), "modelo": art["modelo"], "status": art["status"]}


# ====================================================================== indicadores por regras
# Produto, sentimento, oportunidade, temas e recomendações NÃO são modelos treinados: a base não tem
# rótulos para eles. São regras explícitas de palavras-chave e contexto simples (negação, pergunta,
# problema já resolvido), sem métricas de desempenho.
EXPLICACAO_SCORE = ("Score 0-100 = posição (percentil) da probabilidade BRUTA do classificador para a classe "
                    "'risco de insatisfação' (NPS < 7) entre as previsões fora-da-amostra das transcrições de "
                    "treino: 80 significa que a transcrição pontua acima de 80% delas. É um índice ordinal para "
                    "priorizar leitura — NÃO é probabilidade de churn e as probabilidades NÃO estão calibradas "
                    "(o treino usa class_weight='balanced'). A previsão 'Risco' ocorre quando o score atinge o limiar.")
def _limitacao_modelo(art):
    """Primeira limitação, conforme a origem real dos dados do artefato carregado."""
    if art is None:
        return "Sem modelo carregado: só os indicadores por regras estão disponíveis."
    if art.get("base_sintetica"):
        return "O modelo foi treinado numa base SINTÉTICA: as métricas só demonstram que o código funciona."
    rel, mt = art.get("relatorio_base", {}), art["metricas_teste"]
    texto = (f"O modelo é experimental: treinado com {rel.get('linhas_validas', '?')} reuniões originais com NPS "
             f"({rel.get('risco', '?')} de risco, de poucos clientes); ROC AUC no teste = {mt['ROC AUC']:.2f}")
    if art.get("supera_baseline_teste") is False:
        texto += ", sem superar o baseline. O NPS costuma ser anterior à reunião."
    else:
        texto += ", com muita incerteza (poucos casos de risco)."
    return texto


LIMITACOES = [
    "Alvo derivado do NPS (NPS < 7), não de cancelamentos reais: o sistema NÃO mede churn.",
    "O score não é calibrado; use para priorizar leitura, não como probabilidade.",
    "Produto, sentimento, oportunidade e temas são regras de palavras-chave, sem rótulos nem métricas.",
    "As regras não distinguem quem fala (consultor ou cliente) e tratam negação/resolução de forma simples "
    "(ironia, sarcasmo e contexto longo não são tratados).",
    "Necessário validar com transcrições reais e revisão humana antes de qualquer uso em produção.",
]


def _norm(t):
    """minúsculas, sem acentos."""
    return "".join(c for c in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(c))


_NEGADORES = {"nao", "nunca", "nem", "jamais", "sem", "nenhum", "nenhuma", "nada", "ninguem"}
_JANELA_NEGACAO = 3                       # palavras anteriores, na mesma oração
_CONJ = re.compile(r"\b(mas|porem|contudo|entretanto|todavia)\b")


def _frases(texto):
    """Divide em frases (sem acentos). Devolve [(frase, eh_pergunta)]. Saudações são removidas."""
    norm = _norm(remover_marcacoes(texto))
    norm = re.sub(r"\b(bom dia|boa tarde|boa noite)\b", " ", norm)
    saida = []
    for parte in re.split(r"(?<=[.!?;])\s+|\n+", norm):
        parte = parte.strip()
        if parte:
            saida.append((parte, parte.endswith("?")))
    return saida


def _clausulas(frase):
    limites, ini = [], 0
    for m in _CONJ.finditer(frase):
        limites.append((ini, m.start()))
        ini = m.end()
    limites.append((ini, len(frase)))
    return limites


def _varrer(frase, padrao):
    """Ocorrências de `padrao` com índice da oração e flag de negação (palavras negadoras anteriores)."""
    cl, saida = _clausulas(frase), []
    for m in re.finditer(padrao, frase):
        idx = next((i for i, (a, b) in enumerate(cl) if a <= m.start() <= b), len(cl) - 1)
        antes = re.findall(r"\w+", frase[cl[idx][0]:m.start()])[-_JANELA_NEGACAO:]
        saida.append({"termo": m.group(0), "ini": m.start(), "oracao": idx,
                      "negado": any(w in _NEGADORES for w in antes)})
    return saida


# ----------------------------------------------------------------------- produto/categoria
_PRODUTOS = {
    "Protheus": [r"\bprotheus\b"],
    "RM": [r"\btotvs rm\b", r"\brm (labore|vitae|portal|folha|nucleus|horus|chronus)\b"],
    "Fluig": [r"\bfluig\b"],
}
_CATEGORIAS = {
    "RH / Gestão de pessoas": [r"\brecursos humanos\b", r"\brh\b", r"\bgestao de pessoas\b", r"\badmissao\b",
                               r"\brecrutamento\b", r"\bonboarding\b", r"\bferias\b"],
    "Folha de pagamento": [r"\bfolha(?: de pagamento| salarial)?\b", r"\bholerites?\b",
                           r"\bcontracheques?\b", r"\besocial\b"],
    "Ponto eletrônico": [r"\bponto eletronico\b", r"\bcontrole de ponto\b", r"\brelogio de ponto\b",
                         r"\bmarcacao de ponto\b", r"\bbanco de horas\b", r"\bhoras? extras?\b"],
}
_RM_MAIUSCULO = re.compile(r"\bRM\b")   # 'RM' só conta em maiúsculo (evita ruído de 'rm' minúsculo)


def _contar(norm, padroes_por_nome, extra=None):
    cont, evid = {}, {}
    for nome, pads in padroes_por_nome.items():
        achados = [m.group(0) for p in pads for m in re.finditer(p, norm)]
        if extra and nome in extra:
            achados += extra[nome]
        if achados:
            cont[nome], evid[nome] = len(achados), sorted(set(achados))
    return cont, evid


def identificar_produto(texto):
    """Produto TOTVS (Protheus/RM/Fluig) se citado; senão a categoria de negócio; senão 'Não identificado'."""
    sem_tag = remover_marcacoes(texto) if isinstance(texto, str) else ""
    norm = _norm(sem_tag)
    prod, ev_p = _contar(norm, _PRODUTOS, {"RM": _RM_MAIUSCULO.findall(sem_tag)})
    cat, ev_c = _contar(norm, _CATEGORIAS)
    base = prod or cat
    if not base:
        return {"principal": "Não identificado", "produtos": [], "categorias": [], "evidencias": {}}
    topo = max(base.values())
    lideres = [c for c, n in base.items() if n == topo]
    principal = lideres[0] if len(lideres) == 1 else "Múltiplas: " + ", ".join(lideres)
    return {"principal": principal, "produtos": sorted(prod, key=prod.get, reverse=True),
            "categorias": sorted(cat, key=cat.get, reverse=True), "evidencias": {**ev_p, **ev_c}}


# -------------------------------------------------------------------------------- sentimento
_NEG = (r"\b(problema\w*|erro\w*|falha\w*|falhando|bug\w*|trav\w+|lent[oa]s?|lentidao|instavel|instabilidade|"
        r"indisponivel|fora do ar|reclam\w+|insatisf\w+|decepcion\w+|pessim\w+|ruim|ruins|horrivel|"
        r"demor\w+|atras\w+|frustr\w+|complicad\w+|confus[oa]s?|dificil|dificuldade\w*|quebr\w+|caro|"
        r"inaceitavel|prejuizo\w*|persist\w+|sem solucao|sem resposta|ninguem respond\w+|pendente\w*)\b")
_POS = (r"\b(otim[oa]s?|excelente\w*|perfeitamente|bom|boa|boas|bons|satisfeit[oa]s?|satisfacao|gostei|"
        r"adorei|funciona|funcionam|funcionando|funcionou|rapid[oa]s?|agil|facil|simples|eficiente\w*|"
        r"recomend\w+|parabens|estavel|tranquil[oa]s?|conseguimos|contentes?|elogi\w+)\b")
_RESOLV = (r"\b(resolvid[oa]s?|resolveu|resolveram|resolvemos|resolvi|solucionad[oa]s?|solucionou|"
           r"solucionaram|corrigid[oa]s?|corrigiu|corrigiram|normalizad[oa]s?|normalizou|superad[oa]s?|"
           r"sanad[oa]s?|encerrad[oa]s?|ajustad[oa]s?|ajustaram|regularizad[oa]s?)\b")


_PEDIDO_RESOLUCAO = re.compile(r"\b(preciso|precisamos|precisa|quero|queremos|espero|esperamos|gostaria|"
                               r"gostariamos|necessito)\b[^.!?;]{0,25}?\b(resolvid\w+|solucionad\w+|corrigid\w+|"
                               r"ajustad\w+|normalizad\w+|regularizad\w+)\b")


def _sem_pedidos(frase):
    """'Preciso disso resolvido hoje' é um pedido, não uma resolução: neutraliza o marcador (mesmo tamanho)."""
    for m in list(_PEDIDO_RESOLUCAO.finditer(frase)):
        a, b = m.span(2)
        frase = frase[:a] + " " * (b - a) + frase[b:]
    return frase


def _avaliar_frase(frase):
    """Eventos de sentimento de uma frase: lista de (polaridade, texto, tipo)."""
    negs, poss = _varrer(frase, _NEG), _varrer(frase, _POS)
    ress = _varrer(_sem_pedidos(frase), _RESOLV)
    res_ok = [r for r in ress if not r["negado"]]
    ev, resolvidos, anexou = [], [], False
    for n in negs:
        if n["negado"]:
            ev.append((+1, f"negação + '{n['termo']}' -> positivo", "negacao"))
        elif any(r["oracao"] == n["oracao"] or r["ini"] > n["ini"] for r in res_ok):
            ev.append((+1, f"'{n['termo']}' já resolvido -> positivo", "resolvido"))
            resolvidos.append(n["termo"])
            anexou = True
        else:
            ev.append((-1, f"'{n['termo']}' -> negativo", "problema"))
    for p in poss:
        if p["negado"]:
            ev.append((-1, f"negação + '{p['termo']}' -> negativo", "negacao"))
        else:
            ev.append((+1, f"'{p['termo']}' -> positivo", "positivo"))
    for r in ress:
        if r["negado"]:
            ev.append((-1, f"'{r['termo']}' negado (não resolvido) -> negativo", "nao_resolvido"))
    if res_ok and not anexou and not negs:
        ev.append((+1, f"'{res_ok[0]['termo']}' -> positivo", "positivo"))
    return ev, resolvidos


def analisar_sentimento(texto):
    pos = neg = 0
    evid, resolvidos = [], []
    for frase, pergunta in _frases(texto):
        if pergunta:           # "Teve algum problema?" não expressa sentimento
            continue
        ev, res = _avaliar_frase(frase)
        resolvidos += res
        for pol, txt, _ in ev:
            pos, neg = pos + (pol > 0), neg + (pol < 0)
            evid.append(txt)
    if pos + neg == 0:
        rotulo = "Inconclusivo"
    elif pos and neg and abs(pos - neg) <= 1:
        rotulo = "Misto"
    else:
        rotulo = "Positivo" if pos > neg else "Negativo"
    return {"rotulo": rotulo, "sinais_positivos": pos, "sinais_negativos": neg,
            "problemas_resolvidos": sorted(set(resolvidos)), "evidencias": evid}


# ---------------------------------------------------------------------------- oportunidade
_OPORT = (r"\b(proposta\w*|orcamento\w*|cotacao|contratar|contratacao|contratando|adquir\w+|comprar|"
          r"demonstracao|demo|piloto|prova de conceito|novos? modulos?|expandir|expansao|ampliar|ampliacao|"
          r"upgrade|interess\w+|fechar (o )?contrato|assinar|mais \d* ?usuarios|novas (filiais|unidades|lojas)|"
          r"abrindo mais \d+ (filiais|unidades|lojas)|quero conhecer|gostaria de conhecer|gostaria de ver)\b")
_NAO_OPORT = [  # construções que lembram oportunidade mas não são
    r"propostas? (de|do|da|dos|das) (um |uma |o |a )?(concorrent\w+|outr\w+ (fornecedor|empresa))",
    r"contrat\w+ (mais )?(\d+ )?(colaborador\w*|funcionari\w+|pessoas|profissionais|gente)",
]


def analisar_oportunidade(texto):
    sinais, negados = set(), set()
    for frase, _ in _frases(texto):
        for p in _NAO_OPORT:
            frase = re.sub(p, " ", frase)
        for a in _varrer(frase, _OPORT):
            (negados if a["negado"] else sinais).add(a["termo"])
    n = len(sinais)
    nivel = "Alta" if n >= 3 else "Média" if n == 2 else "Baixa" if n == 1 else "Não identificada"
    return {"nivel": nivel, "sinais": sorted(sinais), "sinais_negados": sorted(negados - sinais)}


# ------------------------------------------------------------------------ sinais de risco/temas
_RISCO_TXT = (r"\b(cancel\w+(?!\s+(\w+\s+){0,2}(reuniao|agenda|encontro|visita|treinamento|horario|chamado))|"
              r"rescind\w+|rescis\w+|desisti\w*|concorrente\w*|trocar de (fornecedor|sistema)|outro fornecedor|"
              r"nao (vou |vamos )?renov\w+|multa|repensar (o )?contrato)\b")
_TEMAS_PROBLEMA = {   # só contam como PROBLEMA se a mesma oração tiver evidência negativa (não negada/resolvida)
    "Integração": r"\b(integra\w*|api|sincroniza\w*|interface\w*|conectar|conexao)\b",
    "Desempenho/instabilidade": r"\b(lent[oa]s?|lentidao|trav\w+|instavel|instabilidade|fora do ar|caiu|caindo)\b",
    "Suporte/atendimento": r"\b(suporte|chamado\w*|atendimento)\b",
}
_DUVIDA_COMERCIAL = (r"\b(preco\w*|quanto custa|desconto\w*|mensalidade|reajuste|cobranca|orcamento\w*|"
                     r"valores|prazo\w*|licenca\w*)\b")


def detectar_sinais(texto):
    """(termos de risco no texto, {tema de problema: termos}, termos de dúvida comercial)."""
    risco, temas, comercial = set(), {}, set()
    for frase, pergunta in _frases(texto):
        for m in re.finditer(_DUVIDA_COMERCIAL, frase):
            comercial.add(m.group(0))
        if pergunta:
            continue
        for a in _varrer(frase, _RISCO_TXT):
            if not a["negado"]:
                risco.add(a["termo"])
        negs, todas = _varrer(frase, _NEG), _varrer(_sem_pedidos(frase), _RESOLV)
        ress = [r for r in todas if not r["negado"]]
        ativas = {n["oracao"] for n in negs if not n["negado"]
                  and not any(r["oracao"] == n["oracao"] or r["ini"] > n["ini"] for r in ress)}
        ativas |= {r["oracao"] for r in todas if r["negado"]}
        for tema, pad in _TEMAS_PROBLEMA.items():
            for a in _varrer(frase, pad):
                if a["oracao"] in ativas and not a["negado"]:
                    temas.setdefault(tema, set()).add(a["termo"])
    return sorted(risco), {k: sorted(v) for k, v in temas.items()}, sorted(comercial)


# ------------------------------------------------------------------------------ modelo / termos
_cache = {}
_PADRAO = object()   # sentinela: artefato=None significa "sem modelo" (explícito)


def obter_artefato(caminho=ARTEFATO_PADRAO):
    """Carrega o artefato uma única vez por processo (nunca retreina)."""
    if str(caminho) not in _cache:
        _cache[str(caminho)] = carregar_artefato(caminho)
    return _cache[str(caminho)]


def score_percentil(art, proba):
    """Percentil (0-100) da probabilidade bruta na distribuição fora-da-amostra do treino (monotônico)."""
    ref = np.asarray(art["oof_referencia"])
    return float(100.0 * np.searchsorted(ref, proba, side="right") / len(ref))


def termos_tfidf(pipe, texto, k=8):
    """Termos do texto presentes no vocabulário TF-IDF ajustado, ordenados pelo peso TF-IDF.
    Se o classificador for linear, informa a direção (coeficiente) e a contribuição (peso x coef)."""
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    x = vec.transform([texto])
    if x.nnz == 0:
        return [], 0
    nomes, coefs = vec.get_feature_names_out(), getattr(clf, "coef_", None)
    idx = sorted(x.nonzero()[1], key=lambda i: -x[0, i])[:k]
    saida = []
    for i in idx:
        item = {"termo": str(nomes[i]), "peso_tfidf": round(float(x[0, i]), 4)}
        if coefs is None:
            item["direcao"] = "n/d (modelo não linear)"
        else:
            c = float(coefs[0][i])
            item["direcao"] = "empurra para risco" if c > 0 else "empurra para seguro"
            item["contribuicao"] = round(c * float(x[0, i]), 4)
        saida.append(item)
    return saida, int(x.nnz)


# ---------------------------------------------------------------------------------- recomendações
_TECNICOS = ("Integração", "Desempenho/instabilidade")


def detectar_conflitos(previsao, sent, oport, sinais_risco, perto_limiar):
    """Divergências entre a previsão do classificador e os indicadores por regras (nunca forçar concordância)."""
    c = []
    if previsao == "Seguro" and (sent["rotulo"] == "Negativo" or sinais_risco):
        c.append("modelo previu seguro, mas as regras acharam sinais negativos")
    if previsao == "Risco" and sent["rotulo"] == "Positivo" and not sinais_risco:
        c.append("modelo previu risco, mas as regras acharam sentimento positivo")
    if previsao == "Risco" and oport["nivel"] == "Alta":
        c.append("risco previsto com oportunidade alta")
    if sent["rotulo"] == "Misto":
        c.append("sentimento misto")
    if perto_limiar:
        c.append("score próximo ao limiar")
    return c


def recomendar(previsao, sent, oport, produto, sinais_risco, temas, comercial, perto_limiar):
    """Regras sobre os sinais acima. Cada ação traz o MOTIVO (sinais que a justificam)."""
    rec, risco_modelo = [], previsao == "Risco"
    problemas_ativos = bool(temas) or sent["rotulo"] == "Negativo" or bool(sinais_risco)
    if risco_modelo or sent["rotulo"] == "Negativo" or sinais_risco:
        motivos = (["modelo previu risco"] if risco_modelo else []) \
            + ([f"sentimento negativo ({sent['sinais_negativos']} sinais)"] if sent["rotulo"] == "Negativo" else []) \
            + ([f"termos de risco: {', '.join(sinais_risco)}"] if sinais_risco else [])
        rec.append({"acao": "Contato proativo do time de sucesso/comercial para investigar a insatisfação.",
                    "motivo": "; ".join(motivos)})
    tecnicos = sum((temas[t] for t in _TECNICOS if t in temas), [])
    if tecnicos:
        nome = produto["principal"] if produto["principal"] != "Não identificado" else None
        rec.append({"acao": "Encaminhar o problema técnico ao suporte/produto" + (f" ({nome})" if nome else "") + ".",
                    "motivo": f"termos técnicos com evidência de problema: {', '.join(tecnicos)}"})
    if "Suporte/atendimento" in temas:
        rec.append({"acao": "Escalar o atendimento/chamados abertos à liderança de suporte.",
                    "motivo": f"termos de suporte com evidência de problema: {', '.join(temas['Suporte/atendimento'])}"})
    if oport["nivel"] in ("Alta", "Média") and sent["rotulo"] != "Negativo":
        rec.append({"acao": "Avaliar oportunidade comercial (proposta ou demonstração).",
                    "motivo": f"sinais de oportunidade ({oport['nivel']}): {', '.join(oport['sinais'])}"})
    elif comercial and not problemas_ativos:
        rec.append({"acao": "Responder à dúvida comercial (preço/prazo) e acompanhar.",
                    "motivo": f"termos: {', '.join(comercial)}"})
    if sent["problemas_resolvidos"] and not problemas_ativos:
        rec.append({"acao": "Registrar a resolução e confirmar a satisfação do cliente em um contato leve.",
                    "motivo": f"problema relatado como já resolvido: {', '.join(sent['problemas_resolvidos'])}"})
    conflito = detectar_conflitos(previsao, sent, oport, sinais_risco, perto_limiar)
    if previsao in ("Inconclusivo", "Indisponível"):
        conflito.append("sem previsão do modelo")
    if sent["rotulo"] == "Inconclusivo" and oport["nivel"] == "Não identificada" and not rec:
        conflito.append("poucos sinais textuais")
    if conflito:
        rec.append({"acao": "Solicitar revisão humana da transcrição.", "motivo": "; ".join(conflito)})
    if not rec:
        rec.append({"acao": "Sem ação específica; manter o acompanhamento padrão.",
                    "motivo": "nenhum sinal de risco, problema ou oportunidade detectado"})
    return rec


# ------------------------------------------------------------------------------ função principal
def analisar_transcricao(texto, artefato=_PADRAO):
    """Transcrição -> Processamento -> Modelo -> Indicadores -> Recomendação.

    Nunca levanta exceção por entrada ruim: devolve {'valido': False, 'erro': ...}.
    artefato: dict do modelo; omitido -> carrega o padrão (uma vez); None -> roda só as regras."""
    ok, motivo = validar_transcricao(texto)
    if not ok:
        return {"valido": False, "erro": motivo}
    art = obter_artefato() if artefato is _PADRAO else artefato
    produto, sent, oport = identificar_produto(texto), analisar_sentimento(texto), analisar_oportunidade(texto)
    sinais_risco, temas, comercial = detectar_sinais(texto)
    out = {"valido": True, "erro": "", "explicacao_score": EXPLICACAO_SCORE, "limitacoes": [_limitacao_modelo(art)] + LIMITACOES,
           "produto": produto, "sentimento": sent, "oportunidade": oport,
           "sinais_risco_texto": sinais_risco, "temas_problema": temas, "termos_comerciais": comercial,
           "indicadores_auxiliares_origem": "regras de palavras-chave (não são modelos treinados)"}
    perto = False
    if art is None:
        out.update(previsao="Indisponível", score=None, proba_bruta=None, limiar=None, termos_tfidf=[], modelo=None,
                   aviso_modelo="Artefato do modelo não encontrado. Rode `python pipeline.py treinar`.")
    else:
        pipe = art["pipeline"]
        termos, n_vocab = termos_tfidf(pipe, texto)
        minimo = art.get("nnz_minimo", 1)
        if n_vocab < minimo:
            out.update(previsao="Inconclusivo", score=None, proba_bruta=None, termos_tfidf=termos,
                       aviso_modelo=(f"Texto curto ou fora do vocabulário do modelo: {n_vocab} termos reconhecidos "
                                     f"(mínimo {minimo}; mediana no treino {art.get('nnz_mediana_treino', 'n/d')}). "
                                     "O classificador não tem evidência suficiente; veja os indicadores por regras."))
        else:
            p = float(pipe.predict_proba([texto])[0, 1])
            sc = score_percentil(art, p)
            perto = abs(sc - score_percentil(art, art["limiar"])) <= JANELA_LIMIAR_SCORE
            out.update(previsao="Risco" if p >= art["limiar"] else "Seguro", score=round(sc, 1),
                       proba_bruta=round(p, 4), termos_tfidf=termos, aviso_modelo="")
        out.update(limiar=round(score_percentil(art, art["limiar"]), 1), limiar_proba_bruta=round(art["limiar"], 4),
                   modelo={"nome": art["modelo"], "status": art["status"], "calibrado": art["calibrado"],
                           "base_sintetica": art.get("base_sintetica", False), "treinado_em": art["treinado_em"],
                           "fonte": art.get("relatorio_base", {}).get("fonte", "n/d"),
                           "arquivo_base": art.get("relatorio_base", {}).get("arquivo", "n/d")})
        avisos = []
        if art.get("base_sintetica"):
            avisos.append("Modelo treinado em base SINTÉTICA: demonstração técnica, sem validade para clientes reais.")
        else:
            avisos.append("Modelo EXPERIMENTAL treinado em poucos dados originais (poucos casos de risco): "
                          "sem evidência de confiabilidade para produção; não prevê churn.")
        out["aviso_modelo"] = " ".join(filter(None, [out["aviso_modelo"]] + avisos))
    out["conflitos"] = detectar_conflitos(out["previsao"], sent, oport, sinais_risco, perto)
    out["recomendacoes"] = recomendar(out["previsao"], sent, oport, produto, sinais_risco, temas, comercial, perto)
    out["revisao_humana"] = any(r["acao"].startswith("Solicitar revisão") for r in out["recomendacoes"])
    return out


# ================================================================ exemplos de demonstração (app)
# Textos CURTOS escritos à mão, sem rótulo e sem métrica: servem para mostrar a saída; o modelo (treinado com
# transcrições longas) tende a se abster nelas, mas as regras funcionam.
EXEMPLOS_DEMO = {'Cliente satisfeito (Fluig)': '[LOCUTOR 1]: Boa tarde, [PESSOA]! Como foi a primeira fase de uso do Fluig '
                               'na [EMPRESA]?\n'
                               '[LOCUTOR 2]: Excelente! O fluxo de aprovação de férias que levava uma semana '
                               'agora fecha em um dia. O time está muito satisfeito.\n'
                               '[LOCUTOR 1]: Que ótimo. E o atendimento do nosso suporte, como avalia?\n'
                               '[LOCUTOR 2]: Foi rápido e atencioso. Quando precisamos de ajuda, resolveram '
                               'tudo no mesmo dia. Recomendo a solução para outras empresas.\n'
                               '[LOCUTOR 1]: Algum ponto de melhoria que queira registrar?\n'
                               '[LOCUTOR 2]: Nada relevante. Está tudo funcionando bem e a equipe aprendeu a '
                               'usar sem dificuldade.\n'
                               '[LOCUTOR 1]: Obrigado pelo retorno, [PESSOA]. Seguimos à disposição.\n'
                               '[LOCUTOR 2]: Eu que agradeço, parabéns ao time de vocês.',
 'Insatisfação com ameaça de cancelamento (Protheus)': '[LOCUTOR 1]: Bom dia, [PESSOA]. Obrigado por separar '
                                                       'um tempo. Como está a operação com o Protheus?\n'
                                                       '[LOCUTOR 2]: Bom dia. Sinceramente, estamos muito '
                                                       'insatisfeitos. O ponto eletrônico trava todos os '
                                                       'dias na hora do fechamento e o suporte demora demais '
                                                       'para responder.\n'
                                                       '[LOCUTOR 1]: Entendo. Há quanto tempo isso vem '
                                                       'acontecendo?\n'
                                                       '[LOCUTOR 2]: Desde a última atualização, há mais de '
                                                       'um mês. Abrimos vários chamados e ninguém resolve de '
                                                       'vez. Meu time perde horas refazendo as marcações na '
                                                       'mão.\n'
                                                       '[LOCUTOR 1]: Imagino o impacto disso na folha de '
                                                       'pagamento.\n'
                                                       '[LOCUTOR 2]: É um prejuízo real. A diretoria já '
                                                       'pediu para avaliar outro fornecedor e, se nada mudar '
                                                       'até o fim do mês, vamos cancelar o contrato.\n'
                                                       '[LOCUTOR 1]: Vou levar isso para a gerência hoje '
                                                       'mesmo e retorno com um plano.\n'
                                                       '[LOCUTOR 2]: Espero que dessa vez seja diferente, '
                                                       'porque a nossa paciência está acabando.',
 'Opinião mista (ponto bom, folha com problemas)': '[LOCUTOR 1]: Boa tarde, [PESSOA]. Como está a '
                                                   'experiência com os módulos até agora?\n'
                                                   '[LOCUTOR 2]: O ponto eletrônico é bom e fácil de usar, '
                                                   'mas a folha de pagamento tem alguns problemas e demora '
                                                   'para fechar.\n'
                                                   '[LOCUTOR 1]: Pode dar um exemplo?\n'
                                                   '[LOCUTOR 2]: Os relatórios são ótimos, só que o '
                                                   'processamento é lento quando fechamos todas as unidades '
                                                   'juntas.\n'
                                                   '[LOCUTOR 1]: E o suporte, tem ajudado?\n'
                                                   '[LOCUTOR 2]: Ajuda, mas às vezes leva dias. Estamos '
                                                   'satisfeitos com o produto, só precisamos de mais '
                                                   'agilidade no atendimento.\n'
                                                   '[LOCUTOR 1]: Vou levar esse ponto para a liderança e '
                                                   'montamos um plano de melhorias.\n'
                                                   '[LOCUTOR 2]: Tudo bem, fico no aguardo.',
 'Oportunidade de expansão': '[LOCUTOR 1]: Oi, [PESSOA]! Como andam os planos da [EMPRESA] para este ano?\n'
                             '[LOCUTOR 2]: Estamos crescendo bastante e abrindo mais cinco unidades. '
                             'Gostaria de uma proposta para ampliar o Protheus com mais usuários.\n'
                             '[LOCUTOR 1]: Que notícia boa! Em quantas pessoas você pensa?\n'
                             '[LOCUTOR 2]: Umas 80 a mais. E também queria uma demonstração do Fluig para o '
                             'time de RH.\n'
                             '[LOCUTOR 1]: Consigo agendar a demonstração para a próxima semana e preparo o '
                             'orçamento da ampliação.\n'
                             '[LOCUTOR 2]: Perfeito. Se o valor couber no orçamento do trimestre, quero '
                             'fechar o contrato ainda neste mês.\n'
                             '[LOCUTOR 1]: Combinado, te envio a proposta até sexta-feira.\n'
                             '[LOCUTOR 2]: Obrigado, fico no aguardo.'}


def exemplos_originais_teste(n=2):
    """Até n transcrições ORIGINAIS (anonimizadas) do hold-out, com nomes neutros (o rótulo não é revelado).
    Permitem ver a saída completa do modelo em textos longos; vazio se a base não estiver disponível."""
    try:
        df, _ = carregar_base()
        _, X_te, _, y_te, _ = dividir_base(df)
    except Exception:
        return {}
    escolha = [X_te[y_te == c].iloc[0] for c in (1, 0) if (y_te == c).any()][:n]
    return {f"Reunião original do conjunto de teste #{i + 1} (anonimizada)": t for i, t in enumerate(escolha)}


# ============================================================ resultados exportados para o app (JSON)
RESULTADOS_PADRAO = PASTA / "resultados_avaliacao.json"


def resumo_descritivo(df, df_bruto=None):
    """Estatísticas DESCRITIVAS da base rotulada (NPS conhecido) e contagens das regras de palavras-chave.
    Nada aqui é previsão do classificador. `df`: saída de carregar_base(); `df_bruto`: carregar_json() (volume com e sem NPS)."""
    y = df["ALVO_RISCO"].to_numpy()

    def agrupar(col):
        g = df.assign(**{col: df[col].fillna("N/D").astype(str)}).groupby(col).agg(
            reunioes=("NOTA_NPS", "size"), nps_medio=("NOTA_NPS", "mean"), risco=("ALVO_RISCO", "sum")).reset_index()
        g = g.rename(columns={col: "grupo"})
        g["pct_risco"] = g["risco"] / g["reunioes"]
        return g.sort_values("reunioes", ascending=False).to_dict("records")

    out = {"nps_media": float(df["NOTA_NPS"].mean()), "clientes_risco": int(df.groupby("CODT")["ALVO_RISCO"].max().sum()),
           "nps_distribuicao": {str(int(k)): int(v) for k, v in df["NOTA_NPS"].value_counts().sort_index().items()},
           "por_segmento": agrupar("NOME_SEGMENTO"), "por_faixa": agrupar("FAIXA_FATURAMENTO_CLIENTE")}
    dias = (pd.to_datetime(df["DT_ULTIMA_PESQUISA"], errors="coerce") - pd.to_datetime(df["DT_MEETING"], errors="coerce")).dt.days.dropna()
    if len(dias):      # o NPS foi coletado antes ou depois da reunião? (limita o que o texto pode explicar)
        out["nps_vs_reuniao"] = {"pct_anterior": float((dias < 0).mean()), "mediana_dias": float(dias.median()), "n": int(len(dias))}
    if df_bruto is not None:
        b = df_bruto.assign(grupo=df_bruto["NOME_SEGMENTO"].fillna("N/D").astype(str), ok=df_bruto["NOTA_NPS"].between(0, 10))
        v = b.groupby("grupo").agg(registros=("ok", "size"), com_nps=("ok", "sum")).reset_index()
        v["sem_nps"] = v["registros"] - v["com_nps"]
        out["volume_segmento"] = v.sort_values("registros", ascending=False).to_dict("records")

    # termos (mesma configuração de TF-IDF do notebook): relevância média e associação descritiva com risco x não risco
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.90, sublinear_tf=True)
    M = vec.fit_transform(df["TEXTO_LIMPO"])
    termos = vec.get_feature_names_out()
    medio = np.asarray(M.mean(axis=0)).ravel()
    pct_doc = np.asarray((M > 0).mean(axis=0)).ravel()
    m_r, m_n = np.asarray(M[y == 1].mean(axis=0)).ravel(), np.asarray(M[y == 0].mean(axis=0)).ravel()
    top = lambda v, k: np.argsort(-v)[:k]
    out["termos"] = {
        "n_risco": int((y == 1).sum()), "n_nao_risco": int((y == 0).sum()),
        "top_tfidf": [{"termo": termos[i], "tfidf_medio": float(medio[i]), "pct_reunioes": float(pct_doc[i])} for i in top(medio, 20)],
        "associados_risco": [{"termo": termos[i], "tfidf_risco": float(m_r[i]), "tfidf_nao_risco": float(m_n[i]),
                              "diferenca": float(m_r[i] - m_n[i])} for i in top(m_r - m_n, 15)],
        "associados_nao_risco": [{"termo": termos[i], "tfidf_risco": float(m_r[i]), "tfidf_nao_risco": float(m_n[i]),
                                  "diferenca": float(m_n[i] - m_r[i])} for i in top(m_n - m_r, 15)]}

    # indicadores por REGRAS aplicados às reuniões rotuladas (sem métrica de acerto: a base não tem rótulo para eles)
    linhas = []
    for texto, alvo in zip(df[COLUNA_TEXTO], y):
        r = analisar_transcricao(texto, artefato=None)
        linhas.append({"alvo": int(alvo), "produto": r["produto"]["principal"],
                       "mencoes": r["produto"]["produtos"] + r["produto"]["categorias"],
                       "sentimento": r["sentimento"]["rotulo"], "oportunidade": r["oportunidade"]["nivel"],
                       "temas": list(r["temas_problema"]), "risco_texto": ["Termos de cancelamento/concorrência"] if r["sinais_risco_texto"] else [],
                       "duvida": ["Dúvida comercial (preço, prazo, licença)"] if r["termos_comerciais"] else []})
    R = pd.DataFrame(linhas)

    def contar(col, explodir):
        s = R[["alvo", col]].explode(col) if explodir else R[["alvo", col]]
        t = s.dropna(subset=[col]).groupby([col, "alvo"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
        t["total"] = t[0] + t[1]
        return [{"item": str(i), "total": int(t.loc[i, "total"]), "nao_risco": int(t.loc[i, 0]), "risco": int(t.loc[i, 1])}
                for i in t.sort_values("total", ascending=False).index]

    out["regras"] = {"n_reunioes": int(len(R)), "n_risco": int((y == 1).sum()), "n_nao_risco": int((y == 0).sum()),
                     "sentimento": contar("sentimento", False), "oportunidade": contar("oportunidade", False),
                     "produto_principal": contar("produto", False), "mencoes": contar("mencoes", True),
                     "temas_problema": contar("temas", True), "sinais_risco_texto": contar("risco_texto", True),
                     "duvida_comercial": contar("duvida", True)}
    return out


def exportar_resultados(df, rel_base, res=None, y_te=None, estab=None, df_bruto=None, caminho=RESULTADOS_PADRAO):
    """Grava `resultados_avaliacao.json`: resultados REAIS já calculados (comparação de modelos no teste e na CV,
    probabilidades do teste, estatísticas descritivas). O app só lê este arquivo: não retreina nada.
    Sem `res`, mantém a seção de modelos do arquivo anterior e atualiza apenas as estatísticas descritivas."""
    caminho = Path(caminho)
    anterior = json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else {}
    dados = {"gerado_em": datetime.now().isoformat(timespec="seconds"), "base": rel_base,
             "descritivo": resumo_descritivo(df, df_bruto)}
    if res is not None:
        dados["modelos"] = [{
            "nome": r["Modelo"], "hiperparametros": {k: str(v) for k, v in r["Hiperparametros"].items()}, "limiar": float(r["Limiar"]),
            "cv": {"AUC média": r["CV AUC media"], "AUC desvio": r["CV AUC desvio"], "PR AUC": r["CV PR AUC"],
                   "Prevalência": r["Prevalencia treino"], "Balanced acc": r["CV Balanced Acc"], "Recall": r["CV Recall"],
                   "% previsto risco": r["CV % previsto risco"]},
            "teste": r["Teste"], "teste_05": r["Teste @0.5"], "auc_ic95": list(r["Teste AUC IC95"]),
            "proba_teste": [round(float(p), 6) for p in res["probas_teste"][r["Modelo"]]]} for r in res["linhas"]]
        dados["y_teste"] = [int(v) for v in np.asarray(y_te)]
        dados["selecionado"] = selecionar_modelo(res, rel_base["fonte"] != "json_original")["modelo"]
        if estab is not None:     # estabilidade_holdout(res, df): só entra se vier da MESMA execução (nunca reaproveita resultado antigo)
            dados["estabilidade"] = estab.rename_axis("Modelo").reset_index().to_dict("records")
    else:
        dados.update({k: anterior[k] for k in ("modelos", "y_teste", "selecionado", "estabilidade") if k in anterior})
    caminho.write_text(json.dumps(dados, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    return caminho


# ===================================================================================== CLI
def _imprimir_resultados(res, sel, rel):
    print("Base:", json.dumps(rel, ensure_ascii=False))
    print("\nVALIDAÇÃO CRUZADA (treino) e limiar:\n" + tabela_cv(res).to_string())
    print("\nTESTE (limiar da política):\n" + tabela_comparativa(res, "Teste").to_string())
    print("\nTESTE (limiar 0,5):\n" + tabela_comparativa(res, "Teste @0.5").to_string())
    print("\nSelecionado:", json.dumps(sel, ensure_ascii=False))


_TEXTO_FUMACA = "\n".join([
    "[LOCUTOR 1]: Bom dia, [PESSOA]. Obrigado por separar um tempo. Como está a operação com o Protheus?",
    "[LOCUTOR 2]: Bom dia. Sinceramente, estamos muito insatisfeitos. O ponto eletrônico trava todos os dias na hora do fechamento e o suporte demora demais para responder.",
    "[LOCUTOR 1]: Entendo. Há quanto tempo isso vem acontecendo?",
    "[LOCUTOR 2]: Desde a última atualização, há mais de um mês. Abrimos vários chamados e ninguém resolve de vez. Meu time perde horas refazendo as marcações na mão.",
    "[LOCUTOR 1]: Imagino o impacto disso na folha de pagamento.",
    "[LOCUTOR 2]: É um prejuízo real. A diretoria já pediu para avaliar outro fornecedor e, se nada mudar até o fim do mês, vamos cancelar o contrato.",
    "[LOCUTOR 1]: Vou levar isso para a gerência hoje mesmo e retorno com um plano.",
    "[LOCUTOR 2]: Espero que dessa vez seja diferente, porque a nossa paciência está acabando."])


def _cmd_testar():
    """Teste de fumaça do artefato e da inferência (as regras são testadas no notebook, Seção 15)."""
    falhas = []

    def conferir(cond, msg):
        print(("OK   " if cond else "FALHA"), msg)
        if not cond:
            falhas.append(msg)

    for entrada in ("", None, "ok", 123):
        r = analisar_transcricao(entrada, artefato=None)
        conferir(r["valido"] is False and bool(r["erro"]), f"entrada inválida rejeitada: {entrada!r}")
    art = carregar_artefato()
    conferir(art is not None, "artefato modelo_risco.joblib carregado")
    if art is not None:
        # texto completo (transcrições reais são longas): primeira transcrição do hold-out, se a base existir
        txt = _TEXTO_FUMACA
        if base_padrao().exists():
            txt = dividir_base(carregar_base()[0])[1].iloc[0]
        r = analisar_transcricao(txt, artefato=art)
        conferir(list(art["pipeline"].classes_) == [0, 1], "probabilidade de risco = coluna da classe 1 (classes_ == [0, 1])")
        conferir(not analisar_transcricao('{"ID_MEETING": "1", "ANON_TRANSCRICAO": "texto"}', artefato=art)["valido"],
                 "JSON/registros colados são rejeitados")
        conferir(r["valido"] and r["previsao"] in ("Risco", "Seguro"), f"previsão devolvida: {r.get('previsao')}")
        conferir(r["score"] is not None and 0 <= r["score"] <= 100, f"score no intervalo 0-100: {r.get('score')}")
        conferir((r["previsao"] == "Risco") == (r["proba_bruta"] >= art["limiar"]), "previsão coerente com o limiar")
        conferir(all(k in r for k in ("produto", "sentimento", "oportunidade", "termos_tfidf", "recomendacoes",
                                      "explicacao_score", "limitacoes")), "saída com todos os campos esperados")
        conferir(all(x["motivo"] for x in r["recomendacoes"]), "toda recomendação traz o motivo")
        curto = analisar_transcricao("[LOCUTOR 2]: Muito ruim, trava toda hora e o suporte não responde.", artefato=art)
        conferir(curto["valido"] and curto["previsao"] == "Inconclusivo" and curto["score"] is None,
                 "texto curto -> 'Inconclusivo' (sem score), regras continuam ativas")
        if base_padrao().exists() and art.get("relatorio_base", {}).get("arquivo") == base_padrao().name:
            v = avaliar_artefato()
            conferir(v["consistente_com_treino"], "artefato reproduz as métricas gravadas no hold-out da base")
    print("\n" + ("Todos os testes de fumaça passaram." if not falhas else f"{len(falhas)} falha(s)."))
    return 1 if falhas else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pipeline Sprint 4 — Data Science (TOTVS Challenge 2026)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gerar", help="gera uma base SINTÉTICA de demonstração (CSV; não é usada no treino padrão)")
    g.add_argument("--n", type=int, default=500)
    g.add_argument("--seed", type=int, default=RANDOM_STATE)
    g.add_argument("--saida", default=str(SINTETICO_PADRAO))
    t = sub.add_parser("treinar", help="treina, compara e salva o artefato")
    t.add_argument("--base", default=None, help="JSON original (padrão) ou CSV; .csv é tratado como demonstração")
    t.add_argument("--artefato", default=str(ARTEFATO_PADRAO))
    a = sub.add_parser("avaliar", help="reavalia candidatos e confere o artefato salvo")
    a.add_argument("--base", default=None)
    a.add_argument("--artefato", default=str(ARTEFATO_PADRAO))
    sub.add_parser("resultados", help="regenera as estatísticas descritivas do app (resultados_avaliacao.json), sem retreinar")
    sub.add_parser("testar", help="teste de fumaça do artefato e da inferência")
    an = sub.add_parser("analisar", help="analisa uma transcrição")
    an.add_argument("texto", nargs="?", help="texto da transcrição (ou use --arquivo)")
    an.add_argument("--arquivo", help="arquivo .txt com a transcrição")
    args = ap.parse_args(argv)

    if args.cmd == "gerar":
        df = salvar_base_sintetica(args.saida, args.n, args.seed)
        print(f"Base SINTÉTICA salva em {args.saida}: {len(df)} linhas, {df.shape[1]} colunas, "
              f"{df[COLUNA_TEXTO].nunique()} transcrições distintas, "
              f"{(df[COLUNA_NPS] < LIMIAR_NPS).mean():.1%} com NPS < {LIMIAR_NPS}.")
    elif args.cmd == "treinar":
        res, sel, rel = treinar_base(args.base, args.artefato)
        _imprimir_resultados(res, sel, rel)
        print("Artefato salvo em:", args.artefato)
    elif args.cmd == "avaliar":
        df, rel = carregar_base(args.base)
        X_tr, X_te, y_tr, y_te, g_tr = dividir_base(df)
        res = avaliar_candidatos(X_tr, y_tr, g_tr, X_te, y_te)
        _imprimir_resultados(res, selecionar_modelo(res, rel["fonte"] != "json_original"), rel)
        print("\nCONTROLE NEGATIVO (rótulos embaralhados):\n" + controle_rotulos_embaralhados(X_tr, y_tr, X_te, y_te).round(3).to_string())
        v = avaliar_artefato(args.base, args.artefato)
        print("\nARTEFATO SALVO no mesmo hold-out:", json.dumps({k: v[k] for k in ("modelo", "status", "consistente_com_treino")}, ensure_ascii=False))
        print({k: round(x, 4) if isinstance(x, float) else x for k, x in v["metricas"].items()})
    elif args.cmd == "resultados":
        df, rel = carregar_base()
        print("Salvo em:", exportar_resultados(df, rel, df_bruto=carregar_json()))
    elif args.cmd == "testar":
        return _cmd_testar()
    elif args.cmd == "analisar":
        texto = Path(args.arquivo).read_text(encoding="utf-8") if args.arquivo else args.texto
        r = analisar_transcricao(texto)
        print(json.dumps({k: v for k, v in r.items() if k not in ("limitacoes", "explicacao_score")},
                         ensure_ascii=False, indent=2))
        if r.get("valido"):
            print("\n" + r["explicacao_score"])
    return 0


if __name__ == "__main__":
    # Reimporta como módulo para que o artefato salvo referencie `pipeline.limpar_texto`
    # (e não `__main__.limpar_texto`), o que permitiria carregá-lo depois no app e no notebook.
    import pipeline as _self
    sys.exit(_self.main())
