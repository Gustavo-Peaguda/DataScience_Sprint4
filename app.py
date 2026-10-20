"""Painel de Inteligência Comercial (protótipo acadêmico) — FIAP Challenge 2026 / TOTVS.

Cinco abas: analisar transcrição (principal), visão geral da base, desempenho dos modelos,
insights de negócio e como funciona. Toda a lógica (limpeza, modelo, regras, scoring, recomendações)
vem de `pipeline.py`; as abas 2 a 4 apenas LEEM resultados já calculados em `resultados_avaliacao.json`
(gerado por `pipeline.py treinar`): nada é retreinado nem recalculado ao abrir o app.
Execute: python -m streamlit run app.py
"""
import html
import json
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import (EXEMPLOS_DEMO, RESULTADOS_PADRAO, analisar_transcricao, exemplos_originais_teste,
                      obter_artefato, score_percentil)

NAVY, TEAL, GRAY, AMBER, CLARO = "#0B2545", "#0E9AA7", "#5B6B7F", "#B45309", "#C9D3DF"
SEM_EXEMPLO = "— nenhum: vou colar minha transcrição —"

st.set_page_config(page_title="Painel de Inteligência Comercial", page_icon="📊", layout="wide")
st.markdown(f"""<style>
.block-container {{max-width: 1150px; padding-top: 1.4rem;}}
.hero {{background: linear-gradient(120deg, {NAVY} 0%, #134074 70%, {TEAL} 140%); color: #fff;
        padding: 20px 28px; border-radius: 14px; margin-bottom: 12px;}}
.hero h1 {{font-size: 1.65rem; margin: 0 0 4px 0; color: #fff;}}
.hero p {{margin: 0; opacity: .9; font-size: .93rem;}}
.aviso {{border-left: 5px solid {TEAL}; background: #EAF6F7; color: {NAVY}; padding: 9px 14px;
         border-radius: 6px; font-size: .88rem; margin-bottom: 10px;}}
.cards {{display: flex; flex-wrap: wrap; gap: 12px; margin: 6px 0 14px 0;}}
.card {{box-sizing: border-box; flex: 1 1 150px; background: #fff; border: 1px solid #D6DEE8; border-radius: 12px; padding: 12px 14px;
        border-top: 4px solid {NAVY};}}
.card.regra {{border-top-color: #9AA7B5;}} .card.base {{border-top-color: {TEAL};}}
.card .tag {{font-size: .66rem; letter-spacing: .06em; text-transform: uppercase; font-weight: 700; color: {NAVY};}}
.card.regra .tag {{color: {GRAY};}} .card.base .tag {{color: {TEAL};}}
.card .rot {{font-size: .82rem; color: {GRAY}; margin-top: 2px;}}
.card .val {{font-size: 1.45rem; font-weight: 700; color: {NAVY}; margin: 4px 0; line-height: 1.2;}}
.card .nota {{font-size: .76rem; color: {GRAY};}}
.val.risco {{color: {AMBER};}} .val.seguro {{color: #0F766E;}} .val.neutro {{color: {GRAY};}}
.sec {{color: inherit; font-size: 1.15rem; font-weight: 700; margin: 22px 0 8px 0; padding-left: 10px; border-left: 4px solid {TEAL};}}
.sub {{color: {GRAY}; font-size: .86rem; margin: -2px 0 8px 0;}}
.selo {{display: inline-block; font-size: .66rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
        padding: 2px 8px; border-radius: 10px; margin-right: 6px; vertical-align: middle;}}
.selo.base {{background: #DDF3F5; color: #0B6E78;}} .selo.assoc {{background: #E8ECF3; color: {NAVY};}}
.selo.modelo {{background: {NAVY}; color: #fff;}} .selo.regra {{background: #EDEFF2; color: {GRAY};}}
.termo {{display: flex; align-items: center; gap: 8px; margin: 3px 0; font-size: .9rem; color: {NAVY};}}
.termo .barra {{height: 8px; border-radius: 4px; background: {TEAL}; min-width: 4px;}}
.termo .dir {{font-size: .75rem; color: {GRAY};}}
.rec {{background: #fff; border: 1px solid #D6DEE8; border-left: 5px solid {TEAL}; border-radius: 8px;
       padding: 10px 14px; margin: 8px 0; color: {NAVY};}}
.rec.revisao {{border-left-color: {AMBER}; background: #FFF8EC;}}
.rec .porque {{font-size: .84rem; color: {GRAY}; margin-top: 3px;}}
.sinal {{font-size: .88rem; color: {NAVY}; margin: 2px 0;}}
.veredito {{border-left: 5px solid {AMBER}; background: #FFF8EC; color: {NAVY}; padding: 12px 16px; border-radius: 6px; margin: 8px 0 12px 0;}}
.veredito b {{color: {AMBER};}}
.fluxo {{display: flex; flex-wrap: wrap; align-items: stretch; gap: 6px; margin: 8px 0 16px 0;}}
.passo {{flex: 1 1 140px; background: #fff; border: 1px solid #D6DEE8; border-top: 4px solid {NAVY}; border-radius: 10px; padding: 10px 12px;}}
.passo.regra {{border-top-color: #9AA7B5;}} .passo.fim {{border-top-color: {TEAL};}}
.passo .n {{font-size: .7rem; font-weight: 700; color: {TEAL}; letter-spacing: .06em;}}
.passo .t {{font-weight: 700; color: {NAVY}; font-size: .95rem; margin: 2px 0;}}
.passo .d {{font-size: .78rem; color: {GRAY};}}
.seta {{align-self: center; color: {TEAL}; font-size: 1.4rem; font-weight: 700;}}
.caixa {{background: #F6F8FB; border: 1px solid #E1E7EF; border-radius: 10px; padding: 12px 14px; color: {NAVY}; font-size: .88rem;}}
.caixa h4 {{margin: 0 0 6px 0; font-size: .95rem; color: {NAVY};}}
.caixa ul {{margin: 4px 0 0 0; padding-left: 18px;}} .caixa li {{margin: 2px 0;}}
button[data-baseweb="tab"] {{font-weight: 600;}}
</style>""", unsafe_allow_html=True)


# ---- estilo da aba "Como funciona" e das abas (cores que funcionam em tema claro e escuro) ----
CSS_EXTRA = """
button[data-baseweb="tab"][aria-selected="true"] {color: __TEAL__ !important;}
div[data-baseweb="tab-highlight"] {background-color: __TEAL__ !important;}
button[data-baseweb="tab"]:focus-visible {outline: 2px solid __TEAL__ !important;}
.fx {background: linear-gradient(120deg, #0B2545 0%, #134074 70%, #0E9AA7 150%); border-radius: 16px; padding: 22px 26px 20px 26px; color: #fff; margin: 4px 0 8px 0;}
.fx-t {font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; opacity: .75; font-weight: 700; margin-bottom: 16px;}
.fx-row {display: grid; grid-template-columns: repeat(6, 1fr);}
.fx-s {position: relative; text-align: center; padding: 0 8px;}
.fx-s:not(:last-child)::after {content: ""; position: absolute; top: 21px; left: calc(50% + 30px); right: calc(-50% + 30px); height: 2px; background: rgba(255,255,255,.35);}
.fx-c {width: 44px; height: 44px; border-radius: 50%; margin: 0 auto 10px auto; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.05rem; background: #fff; color: #0B2545;}
.fx-s.m .fx-c {background: __TEAL__; color: #fff; box-shadow: 0 0 0 5px rgba(14,154,167,.35);}
.fx-s.r .fx-c {background: transparent; color: #fff; border: 2px dashed rgba(255,255,255,.85);}
.fx-n {font-weight: 700; font-size: .95rem; line-height: 1.2; margin-bottom: 4px;}
.fx-d {font-size: .76rem; opacity: .8; line-height: 1.35;}
.fx-g {display: inline-block; font-size: .6rem; letter-spacing: .08em; text-transform: uppercase; font-weight: 700; padding: 1px 8px; border-radius: 10px; margin-bottom: 5px;}
.fx-s.m .fx-g {background: __TEAL__; color: #fff;} .fx-s.r .fx-g {border: 1px solid rgba(255,255,255,.7);}
.leg {display: flex; flex-wrap: wrap; gap: 18px; font-size: .78rem; opacity: .8; margin: 0 0 6px 4px;}
.leg i {display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 6px; vertical-align: -1px;}
.kc-grid, .vs-grid, .duo {display: grid; gap: 14px;}
.kc-grid {grid-template-columns: repeat(3, 1fr);} .duo {grid-template-columns: repeat(2, 1fr);}
.vs-grid {grid-template-columns: 1fr 44px 1fr; align-items: stretch;}
.kc, .vs {border: 1px solid rgba(128,128,128,.3); border-top: 4px solid __TEAL__; border-radius: 12px; padding: 16px 18px; background: rgba(128,128,128,.07);}
.kc .k, .vs .k {font-size: .66rem; letter-spacing: .09em; font-weight: 700; text-transform: uppercase; color: __TEAL__;}
.kc h4, .vs h4 {margin: 3px 0 10px 0; font-size: 1.05rem; padding: 0;}
.kc p {font-size: .85rem; opacity: .85; margin: 10px 0 0 0; line-height: 1.45;}
.pills {display: flex; flex-wrap: wrap; gap: 6px;}
.pill {font-size: .78rem; font-weight: 600; padding: 4px 12px; border-radius: 14px; color: #fff;}
.pill.r {background: __AMBER__;} .pill.s {background: #0F766E;}
.prop {height: 9px; border-radius: 5px; background: #0F766E; margin: 12px 0 4px 0; overflow: hidden;}
.prop div {height: 100%; background: __AMBER__;}
.cap {font-size: .74rem; opacity: .7;}
.chips {display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 4px;}
.chip {font-size: .8rem; padding: 3px 11px; border-radius: 14px; border: 1px solid rgba(128,128,128,.45);}
.chip.alto {background: __TEAL__; border-color: __TEAL__; color: #fff; font-weight: 700; font-size: .9rem;}
.chip.baixo {opacity: .55; font-size: .72rem;}
.gauge {position: relative; height: 12px; border-radius: 6px; margin: 26px 0 6px 0;}
.gauge .mk {position: absolute; top: -22px; transform: translateX(-50%); font-size: .68rem; font-weight: 700; text-align: center; white-space: nowrap;}
.gauge .mk::after {content: ""; display: block; width: 2px; height: 30px; background: currentColor; margin: 1px auto 0 auto;}
.esc {display: flex; justify-content: space-between; font-size: .7rem; opacity: .65;}
.vs {border-top-color: #2F6DB5;} .vs.reg {border-top-color: #8493A5;}
.vs.reg .k {color: #8493A5;} .vs .k {color: #4C8BD6;}
.vs-x {align-self: center; width: 40px; height: 40px; border-radius: 50%; background: #0B2545; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 700; border: 2px solid rgba(255,255,255,.25);}
.rw {display: flex; gap: 12px; padding: 7px 0; border-bottom: 1px solid rgba(128,128,128,.22); font-size: .86rem;}
.rw:last-child {border-bottom: none;}
.rw b {flex: 0 0 96px; font-size: .74rem; font-weight: 600; opacity: .65; text-transform: uppercase; letter-spacing: .04em; padding-top: 2px;}
.nt {text-align: center; font-size: .82rem; opacity: .8; margin: 10px 0 0 0;}
.il {list-style: none; padding: 0; margin: 0;}
.il li {display: flex; gap: 12px; padding: 8px 0; font-size: .87rem; line-height: 1.4; border-bottom: 1px solid rgba(128,128,128,.18);}
.il li:last-child {border-bottom: none;}
.il .ic {flex: 0 0 24px; height: 24px; border-radius: 50%; text-align: center; font-weight: 700; font-size: .8rem; line-height: 24px; color: #fff;}
.il.lim .ic {background: __AMBER__;} .il.ok .ic {background: #0F766E;}
.il b {display: block;}
.lim-c {border-top-color: __AMBER__ !important;} .ok-c {border-top-color: #0F766E !important;}
@media (max-width: 900px) {.kc-grid, .vs-grid, .duo {grid-template-columns: 1fr;} .fx-row {grid-template-columns: repeat(2, 1fr); row-gap: 20px;} .fx-s::after {display: none;} .vs-x {margin: 0 auto;}}
""".replace("__TEAL__", TEAL).replace("__AMBER__", AMBER)
st.markdown(f"<style>{CSS_EXTRA}</style>", unsafe_allow_html=True)


# ============================================================================== utilidades
@st.cache_resource
def carregar_modelo():
    """Carrega modelo_risco.joblib uma única vez por processo; nunca retreina."""
    return obter_artefato()


@st.cache_data
def exemplos():
    return {**EXEMPLOS_DEMO, **exemplos_originais_teste()}


@st.cache_data
def carregar_resultados():
    """Lê os resultados REAIS já calculados (comparação de modelos, estatísticas da base). None se o arquivo não existir."""
    try:
        with open(RESULTADOS_PADRAO, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def br(x, n=2):
    """Número no padrão brasileiro (vírgula decimal, ponto de milhar)."""
    return f"{x:,.{n}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, n=1):
    return br(100 * x, n) + "%"


def card(rotulo, valor, nota="", origem="modelo", classe=""):
    tag = {"modelo": "Classificador experimental", "regra": "Regras de palavras-chave", "base": "Observado na base"}[origem]
    css = {"modelo": "", "regra": "regra", "base": "base"}[origem]
    return (f'<div class="card {css}"><div class="tag">{tag}</div>'
            f'<div class="rot">{html.escape(rotulo)}</div><div class="val {classe}">{html.escape(str(valor))}</div>'
            f'<div class="nota">{html.escape(nota)}</div></div>')


def cartoes(itens):
    st.markdown(f'<div class="cards">{"".join(itens)}</div>', unsafe_allow_html=True)


def secao(titulo, selo=None, legenda=None):
    rotulos = {"base": "Observado na base", "assoc": "Associação textual", "modelo": "Classificador", "regra": "Regras de palavras-chave"}
    s = f'<span class="selo {selo}">{rotulos[selo]}</span>' if selo else ""
    st.markdown(f'<div class="sec">{s}{html.escape(titulo)}</div>', unsafe_allow_html=True)
    if legenda:
        st.markdown(f'<div class="sub">{legenda}</div>', unsafe_allow_html=True)


def lista_sinais(titulo, itens):
    if itens:
        st.markdown(f"**{titulo}:** " + "; ".join(html.escape(str(i)) for i in itens))


def nova_fig(w=9.5, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.set_axisbelow(True)
    return fig, ax


def mostrar(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def tabela(df, formatos=None):
    """Tabela compacta e organizada. formatos: {coluna: 'printf'} para colunas numéricas."""
    cfg = {c: st.column_config.NumberColumn(c, format=f) for c, f in (formatos or {}).items()}
    st.dataframe(df, hide_index=True, column_config=cfg)


def sem_resultados(o_que):
    st.info(f"{o_que} não está disponível: o arquivo `resultados_avaliacao.json` não foi encontrado. "
            "Ele é gerado por `python pipeline.py treinar` (resultados) ou `python pipeline.py resultados` (estatísticas da base).")


def _usar_exemplo():
    nome = st.session_state.get("exemplo", SEM_EXEMPLO)
    if nome != SEM_EXEMPLO:
        st.session_state["texto"] = exemplos()[nome]
        st.session_state.pop("analise", None)


def _limpar():
    st.session_state["texto"] = ""
    st.session_state["exemplo"] = SEM_EXEMPLO
    st.session_state.pop("analise", None)


# ============================================================================== aba 1 — analisar transcrição
def aba_analisar(art):
    st.session_state.setdefault("texto", "")
    secao("1. Transcrição")
    st.selectbox("Exemplo opcional (carrega no campo abaixo; você pode editar ou apagar)",
                 [SEM_EXEMPLO] + list(exemplos()), key="exemplo", on_change=_usar_exemplo)
    texto = st.text_area("Cole aqui UMA transcrição completa", key="texto", height=220,
                         placeholder="[LOCUTOR 1]: Bom dia! Como está o sistema?\n[LOCUTOR 2]: ...")
    b1, b2, _ = st.columns([1, 1, 4])
    analisar = b1.button("Analisar transcrição", type="primary")
    b2.button("Limpar", on_click=_limpar)
    st.caption("Reuniões reais têm milhares de palavras. Trechos curtos (como os exemplos) deixam a previsão do modelo "
               "'Inconclusivo'; os indicadores por regras continuam valendo.")

    if analisar:
        st.session_state["analise"] = analisar_transcricao(texto, artefato=art)

    r = st.session_state.get("analise")
    if r is None:
        return
    if not r["valido"]:
        st.warning(r["erro"])
        return

    secao("2. Resultado")
    prev = r["previsao"]
    classe = {"Risco": "risco", "Seguro": "seguro"}.get(prev, "neutro")
    s, o = r["sentimento"], r["oportunidade"]
    score = "—" if r["score"] is None else f'{r["score"]:.0f} / 100'
    nota_score = "índice de priorização, não é probabilidade" if r["score"] is not None else "sem previsão"
    cartoes([card("Previsão experimental", prev, "risco de insatisfação (NPS < 7)", classe=classe),
             card("Score", score, nota_score),
             card("Produto / categoria", r["produto"]["principal"], "citado na conversa", "regra"),
             card("Sentimento", s["rotulo"], f'+{s["sinais_positivos"]} / −{s["sinais_negativos"]} sinais', "regra"),
             card("Oportunidade comercial", o["nivel"], f'{len(o["sinais"])} sinal(is)', "regra")])
    if prev == "Inconclusivo" and r.get("aviso_modelo"):
        st.info(r["aviso_modelo"].split(" Modelo EXPERIMENTAL")[0])
    if r["conflitos"]:
        st.warning("**Os indicadores divergem — recomenda-se revisão humana.** " + "; ".join(r["conflitos"]) + ".")

    secao("3. Termos e sinais encontrados")
    ct, cs = st.columns(2)
    with ct:
        st.markdown("**Termos do texto que o modelo usa** (TF-IDF)")
        termos = r["termos_tfidf"][:6]
        if termos:
            maximo = max(t["peso_tfidf"] for t in termos) or 1
            linhas = []
            for t in termos:
                dirc = {"empurra para risco": "→ associado a risco", "empurra para seguro": "→ associado a seguro"}.get(t["direcao"], "")
                largura = int(120 * t["peso_tfidf"] / maximo)
                linhas.append(f'<div class="termo"><span style="min-width:150px">{html.escape(t["termo"])}</span>'
                              f'<span class="barra" style="width:{largura}px"></span><span class="dir">{dirc}</span></div>')
            st.markdown("".join(linhas), unsafe_allow_html=True)
            st.caption("A barra mostra a importância do termo neste texto; a seta, para onde o modelo o associa. São pistas, não causas.")
        else:
            st.write("Nenhum termo do vocabulário do modelo.")
    with cs:
        st.markdown("**Sinais por regras de palavras-chave**")
        lista_sinais("Sentimento", s["evidencias"][:4])
        lista_sinais("Já resolvido", s["problemas_resolvidos"])
        lista_sinais("Termos de risco", r["sinais_risco_texto"])
        lista_sinais("Temas de problema", list(r["temas_problema"]))
        lista_sinais("Oportunidade", o["sinais"][:5])
        lista_sinais("Oportunidade negada", o["sinais_negados"])
        lista_sinais("Produtos/categorias", r["produto"]["produtos"] + r["produto"]["categorias"])
        if not (s["evidencias"] or r["sinais_risco_texto"] or r["temas_problema"] or o["sinais"]):
            st.write("Poucos sinais textuais encontrados.")

    secao("4. Recomendações")
    for rec in r["recomendacoes"]:
        extra = " revisao" if rec["acao"].startswith("Solicitar revisão") else ""
        st.markdown(f'<div class="rec{extra}"><b>{html.escape(rec["acao"])}</b>'
                    f'<div class="porque">Por quê: {html.escape(rec["motivo"])}</div></div>', unsafe_allow_html=True)

    with st.expander("Detalhes técnicos"):
        m = r.get("modelo")
        if m:
            st.write(f'Modelo: **{m["nome"]}** · status: `{m["status"]}` · treinado em {m["treinado_em"]} · '
                     f'base: `{m.get("arquivo_base", "n/d")}` ({"sintética" if m["base_sintetica"] else "dados originais"})')
            st.write(f'Probabilidade bruta: {r["proba_bruta"]} · limiar da probabilidade: {r["limiar_proba_bruta"]} · '
                     f'limiar no score: {r["limiar"]}')
            mt = art["metricas_teste"]
            st.write(f'Desempenho no teste (poucos casos de risco, muito incerto): ROC AUC {mt["ROC AUC"]:.2f}, '
                     f'recall {mt["Recall"]:.2f}, precision {mt["Precision"]:.2f}, balanced accuracy {mt["Balanced Acc"]:.2f}.')
        st.caption(r["explicacao_score"])
    with st.expander("Limitações"):
        for item in r["limitacoes"]:
            st.markdown(f"- {item}")


# ============================================================================== aba 2 — visão geral da base
NOMES = {"SAUDE": "Saúde", "SERVICOS": "Serviços", "DISTRIBUICAO": "Distribuição", "LOGISTICA": "Logística", "AGROINDUSTRIA": "Agroindústria",
         "CONSTRUCAO E PROJETOS": "Construção e projetos", "N/D": "Sem segmento informado"}


NOMES_MODELO = {"Regressao Logistica": "Regressão Logística"}


def nm(nome):
    return NOMES_MODELO.get(nome, nome)


def nome_segmento(s):
    return NOMES.get(str(s), str(s).capitalize())


def _faixa_curta(nome):
    """Rótulo de faixa quebrado em linhas; '$' é escapado porque o matplotlib o interpreta como fórmula."""
    return "\n".join(textwrap.wrap(str(nome), 34)).replace("$", r"\$") or "N/D"


def aba_base(R):
    if not R or "descritivo" not in R:
        sem_resultados("A visão geral da base")
        return
    D, B = R["descritivo"], R["base"]
    total, ausente = B["linhas_originais"], B["nps_invalido_ou_nulo"]
    valido, n, risco, seguro = total - ausente, B["linhas_validas"], B["risco"], B["nao_risco"]
    st.markdown('<div class="sub">Estatísticas <b>observadas</b> nos dados originais (NPS informado pelo cliente). '
                'Não são previsões do modelo; "risco" aqui é a regra NPS &lt; 7 (insatisfação), <b>não</b> cancelamento real.</div>',
                unsafe_allow_html=True)
    itens = [card("Transcrições no arquivo", br(total, 0), "reuniões anonimizadas", "base"),
             card("Com NPS informado", br(valido, 0), f"{pct(valido / total)} do arquivo" + (f"; {n} após remover {valido - n} repetida(s)" if valido != n else ""), "base"),
             card("Sem NPS", br(ausente, 0), f"{pct(ausente / total)}; sem rótulo, fora da análise de risco", "base"),
             card("Risco (NPS < 7)", br(risco, 0), f"{pct(risco / n)} das {n} reuniões rotuladas", "base", "risco"),
             card("Sem risco (NPS ≥ 7)", br(seguro, 0), f"{pct(seguro / n)} das reuniões rotuladas", "base", "seguro"),
             card("NPS médio", br(D["nps_media"], 2), "amostra rotulada (não é a base inteira)", "base")]
    if D.get("nps_vs_reuniao"):
        v = D["nps_vs_reuniao"]
        itens.append(card("NPS anterior à reunião", pct(v["pct_anterior"], 0), f"mediana de {br(abs(v['mediana_dias']), 0)} dias antes", "base"))
    cartoes(itens)
    if D.get("nps_vs_reuniao"):
        st.caption("Como a pesquisa de NPS costuma ser anterior à reunião, o rótulo descreve a satisfação do cliente antes da conversa, "
                   "o que limita o quanto o texto pode explicá-lo. Os "
                   f"{n} registros rotulados vêm de {B['clientes_distintos']} clientes (reuniões do mesmo cliente repetem o rótulo).")

    # ---- distribuição das notas
    secao("Distribuição das notas NPS", "base", "Reuniões rotuladas, por nota. Cor âmbar = NPS < 7 (classe de risco).")
    dist = {int(k): v for k, v in D["nps_distribuicao"].items()}
    notas = list(range(11))
    cont = [dist.get(k, 0) for k in notas]
    fig, ax = nova_fig(9.5, 3.4)
    ax.bar(notas, cont, color=[AMBER if k < 7 else TEAL for k in notas], edgecolor="white")
    for k, v in zip(notas, cont):
        ax.text(k, v + max(cont) * 0.01, str(v), ha="center", fontsize=9, color=NAVY)
    ax.axvline(6.5, color=NAVY, ls="--", lw=1)
    ax.set_xticks(notas); ax.set_xlabel("NOTA_NPS"); ax.set_ylabel("reuniões")
    ax.yaxis.grid(True, color="#E3E8EF")
    mostrar(fig)
    with st.expander("Ver valores exatos"):
        tabela(pd.DataFrame({"Nota": notas, "Reuniões": cont, "% das rotuladas": [100 * c / n for c in cont]}), {"% das rotuladas": "%.1f"})

    # ---- volume por segmento
    if D.get("volume_segmento"):
        secao("Volume de reuniões por segmento", "base", "Todas as transcrições do arquivo: com NPS (usadas na análise) e sem NPS (sem rótulo).")
        vol = pd.DataFrame(D["volume_segmento"]).sort_values("registros")
        nomes = [nome_segmento(g) for g in vol["grupo"]]
        fig, ax = nova_fig(9.5, 3.9)
        ax.barh(nomes, vol["com_nps"], color=NAVY, label="com NPS")
        ax.barh(nomes, vol["sem_nps"], left=vol["com_nps"], color=CLARO, label="sem NPS")
        for y, (a, b) in enumerate(zip(vol["com_nps"], vol["sem_nps"])):
            ax.text(a + b + max(vol["registros"]) * 0.01, y, f"{a + b}", va="center", fontsize=9, color=NAVY)
        ax.set_xlabel("registros"); ax.legend(frameon=False, loc="lower right"); ax.xaxis.grid(True, color="#E3E8EF")
        mostrar(fig)
        t = pd.DataFrame({"Segmento": nomes, "Registros": vol["registros"].to_numpy(), "Com NPS": vol["com_nps"].to_numpy(),
                          "Sem NPS": vol["sem_nps"].to_numpy()}).iloc[::-1]
        t["% com NPS"] = 100 * t["Com NPS"] / t["Registros"]
        tabela(t, {"% com NPS": "%.1f"})

    # ---- NPS por segmento / faixa
    def bloco_nps(chave, titulo, legenda, ordenar_por_nome=False):
        g = pd.DataFrame(D[chave])
        grande, pequeno = g[g["reunioes"] >= 10], g[g["reunioes"] < 10]
        # barras horizontais desenham o 1º item embaixo: faixas em ordem natural ficam com a menor no topo
        grande = grande.sort_values("grupo", ascending=False) if ordenar_por_nome else grande.sort_values("nps_medio")
        secao(titulo, "base", legenda)
        media = D["nps_media"]
        fig, ax = nova_fig(9.5, max(3.0, 0.42 * len(grande) + 1.2))
        rot = [_faixa_curta(x) if chave == "por_faixa" else nome_segmento(x) for x in grande["grupo"]]
        ax.barh(rot, grande["nps_medio"], color=[TEAL if v >= media else "#7A8CA3" for v in grande["nps_medio"]], edgecolor="white")
        for y, (v, nn) in enumerate(zip(grande["nps_medio"], grande["reunioes"])):
            ax.text(v + 0.05, y, f"{br(v, 1)}  (n={nn})", va="center", fontsize=9, color=NAVY,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1))
        ax.axvline(media, color=NAVY, ls="--", lw=1, label=f"média geral {br(media, 2)}")
        ax.set_xlim(0, 10.8); ax.set_xlabel("NPS médio"); ax.legend(frameon=False, loc="lower right"); ax.xaxis.grid(True, color="#E3E8EF")
        mostrar(fig)
        t = grande.iloc[::-1][["grupo", "reunioes", "nps_medio", "pct_risco"]].copy()
        t.columns = ["Grupo", "Reuniões", "NPS médio", "% com NPS < 7"]
        if chave == "por_segmento":
            t["Grupo"] = [nome_segmento(x) for x in t["Grupo"]]
        t["% com NPS < 7"] = 100 * t["% com NPS < 7"]
        tabela(t, {"NPS médio": "%.2f", "% com NPS < 7": "%.1f"})
        if len(pequeno):
            st.caption(f"Omitidos por terem menos de 10 reuniões (média instável): {len(pequeno)} grupo(s), {int(pequeno['reunioes'].sum())} reuniões.")

    bloco_nps("por_segmento", "NPS médio por segmento", "Verde-azulado = acima da média geral. Diferenças descritivas; com poucas reuniões podem ser acaso.")
    bloco_nps("por_faixa", "NPS médio por faixa de faturamento do cliente", "Faixas em ordem crescente de faturamento.", ordenar_por_nome=True)


# ============================================================================== aba 3 — desempenho dos modelos
def aba_modelos(R, art):
    if not R or "modelos" not in R:
        sem_resultados("A comparação de modelos")
        return
    M, y, sel = R["modelos"], np.array(R["y_teste"]), R["selecionado"]
    reais = [m for m in M if not m["nome"].startswith("Baseline")]
    aucs = [m["teste"]["ROC AUC"] for m in reais]
    cv_max = max(m["cv"]["AUC média"] for m in reais)
    n_te, r_te = len(y), int(y.sum())
    prev_te = r_te / n_te
    status = (art or {}).get("status", "")

    # ---- veredito (calculado a partir dos resultados)
    superior = [m for m in reais if m["auc_ic95"][0] > 0.5]
    if superior:
        corpo = (f"Os modelos {', '.join(m['nome'] for m in superior)} tiveram IC 95% do ROC AUC acima de 0,5 no teste, mas o teste tem só "
                 f"{r_te} casos de risco: trate o resultado como preliminar.")
        titulo = "Há indício de desempenho acima do acaso, ainda preliminar"
    else:
        titulo = "Nenhum modelo demonstrou desempenho superior ao acaso ou ao baseline"
        corpo = (f"No conjunto de teste, o ROC AUC dos modelos treinados ficou entre <b>{br(min(aucs), 2)}</b> e <b>{br(max(aucs), 2)}</b> "
                 f"(acaso e baseline = 0,50) e o intervalo de confiança de todos inclui 0,5 ou fica abaixo. Na validação cruzada do treino o AUC chegou a "
                 f"{br(cv_max, 2)}, mas isso <b>não se confirmou</b> no teste independente. O modelo é mantido apenas como <b>candidato experimental</b>.")
    st.markdown(f'<div class="veredito"><b>{titulo}.</b> {corpo}</div>', unsafe_allow_html=True)

    treino_n = R["base"]["linhas_validas"] - n_te
    treino_r = R["base"]["risco"] - r_te
    cartoes([card("Teste (hold-out)", f"{n_te} reuniões", f"{r_te} de risco ({pct(prev_te)}); não usado em nenhuma escolha", "base"),
             card("Treino", f"{treino_n} reuniões", f"{treino_r} de risco ({pct(treino_r / treino_n)})", "base"),
             card("Modelo escolhido", nm(sel), f"status: {status or 'n/d'}", "modelo"),
             card("PR AUC de referência", br(prev_te, 3), "= prevalência do risco no teste (acaso)", "base")])

    modo = st.radio("Limiar de decisão usado nas tabelas e matrizes",
                    ["Política de negócio (recall máximo com especificidade ≥ 50%, definida no treino)", "Padrão 0,5"], horizontal=True, key="modo_limiar")
    chave = "teste" if modo.startswith("Política") else "teste_05"
    limiar = lambda m: m["limiar"] if chave == "teste" else 0.5

    # ---- tabela comparativa
    secao("Tabela comparativa no conjunto de teste", "modelo", "ROC AUC e PR AUC não dependem do limiar; as demais métricas mudam com ele.")
    linhas = []
    for m in M:
        t = m[chave]
        linhas.append({"Modelo": nm(m["nome"]) + (" ★" if m["nome"] == sel else ""), "Limiar": limiar(m), "Accuracy": t["Accuracy"],
                       "Precision": t["Precision"], "Recall": t["Recall"], "F1": t["F1"], "Balanced acc.": t["Balanced Acc"],
                       "ROC AUC": t["ROC AUC"], "PR AUC": t["PR AUC"], "% previsto risco": 100 * t["% previsto risco"]})
    comp = pd.DataFrame(linhas)
    tabela(comp, {c: "%.3f" for c in comp.columns[1:-1]} | {"% previsto risco": "%.1f"})
    st.caption("★ = modelo escolhido pela regra fixada antes de olhar o teste. Precision e F1 do baseline são 0 porque ele nunca prevê risco.")

    # ---- ROC AUC e PR AUC
    secao("ROC AUC e PR AUC", "modelo", "Barras de erro: IC 95% por bootstrap do ROC AUC (poucos casos de risco → muita incerteza).")
    nomes = [nm(m["nome"]) for m in M]
    cores = [GRAY if n.startswith("Baseline") else (NAVY if n == sel else TEAL) for n in nomes]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=9); ax.set_axisbelow(True); ax.yaxis.grid(True, color="#E3E8EF")
    pos = np.arange(len(M))
    roc = np.array([m["teste"]["ROC AUC"] for m in M])
    lo = np.array([m["auc_ic95"][0] for m in M]); hi = np.array([m["auc_ic95"][1] for m in M])
    a1.bar(pos, roc, color=cores, edgecolor="white")
    a1.errorbar(pos, roc, yerr=[np.maximum(roc - lo, 0), np.maximum(hi - roc, 0)], fmt="none", ecolor=NAVY, capsize=3, lw=1)
    a1.axhline(0.5, color=AMBER, ls="--", lw=1.2, label="acaso (0,50)")
    for p, v in zip(pos, roc):
        a1.text(p, 0.02, br(v, 2), ha="center", fontsize=9, color="white", fontweight="bold")
    a1.set_ylim(0, 1); a1.set_title("ROC AUC no teste", fontsize=11, color=NAVY); a1.legend(frameon=False, fontsize=9, loc="upper right")
    prc = np.array([m["teste"]["PR AUC"] for m in M])
    a2.bar(pos, prc, color=cores, edgecolor="white")
    a2.axhline(prev_te, color=AMBER, ls="--", lw=1.2, label=f"prevalência = acaso ({br(prev_te, 3)})")
    for p, v in zip(pos, prc):
        a2.text(p, v + 0.005, br(v, 3), ha="center", fontsize=9, color=NAVY)
    a2.set_ylim(0, max(0.4, prc.max() * 1.5)); a2.set_title("PR AUC no teste", fontsize=11, color=NAVY); a2.legend(frameon=False, fontsize=9, loc="upper right")
    for ax in (a1, a2):
        ax.set_xticks(pos); ax.set_xticklabels(["\n".join(textwrap.wrap(n, 12)) for n in nomes], fontsize=8.5)
    mostrar(fig)

    # ---- matrizes de confusão
    secao("Matrizes de confusão (teste)", "modelo", "Linhas = classe real; colunas = previsão. Os números são contagens de reuniões.")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    axes = axes.ravel()
    for ax, m in zip(axes, M):
        t = m[chave]
        mat = np.array([[t["TN"], t["FP"]], [t["FN"], t["TP"]]])
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(mat.max(), 1))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=14, fontweight="bold",
                        color="white" if mat[i, j] > mat.max() * 0.55 else NAVY)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Seguro", "Risco"], fontsize=9); ax.set_yticks([0, 1]); ax.set_yticklabels(["Seguro", "Risco"], fontsize=9)
        ax.set_xlabel("previsto", fontsize=9); ax.set_ylabel("real", fontsize=9)
        ax.set_title(f"{nm(m['nome'])}\n(limiar {br(limiar(m), 3)})", fontsize=10, color=NAVY)
    for ax in axes[len(M):]:
        ax.axis("off")
    mostrar(fig)

    # ---- distribuição das previsões
    secao("Distribuição das previsões de risco", "modelo",
          "Limiar (valor de probabilidade) e % previsto como risco são coisas diferentes: mudar o limiar altera a %, não a capacidade de discriminar.")
    pct_prev = [100 * m[chave]["% previsto risco"] for m in M]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=9); ax.set_axisbelow(True)
    a1.bar(pos, pct_prev, color=cores, edgecolor="white")
    a1.axhline(100 * prev_te, color=AMBER, ls="--", lw=1.2, label=f"risco real no teste ({pct(prev_te)})")
    for p, v in zip(pos, pct_prev):
        a1.text(p, v + 1, f"{br(v, 1)}%", ha="center", fontsize=9, color=NAVY)
    a1.set_xticks(pos); a1.set_xticklabels(["\n".join(textwrap.wrap(n, 12)) for n in nomes], fontsize=8.5)
    a1.set_ylim(0, 100); a1.set_ylabel("% das reuniões previstas como risco"); a1.legend(frameon=False, fontsize=9); a1.yaxis.grid(True, color="#E3E8EF")
    a1.set_title("% prevista como risco", fontsize=11, color=NAVY)
    ms = next(m for m in M if m["nome"] == sel)
    p = np.array(ms["proba_teste"])
    bins = np.linspace(p.min(), p.max(), 21)
    a2.hist(p[y == 0], bins=bins, color=TEAL, alpha=0.75, label="real: seguro")
    a2.hist(p[y == 1], bins=bins, color=AMBER, alpha=0.85, label="real: risco")
    a2.axvline(limiar(ms), color=NAVY, ls="--", lw=1.2, label=f"limiar {br(limiar(ms), 3)}")
    a2.set_title(f"P(risco) bruta — {nm(sel)}", fontsize=11, color=NAVY); a2.set_xlabel("probabilidade bruta (não calibrada)"); a2.set_ylabel("reuniões")
    a2.legend(frameon=False, fontsize=9)
    mostrar(fig)
    st.caption("O DummyClassifier (strategy='prior') atribui a todas as reuniões a prevalência do treino (~0,16): como 0,16 < 0,5, "
               "nenhuma é prevista como risco (0%). Isso é esperado, não um erro. A sobreposição das duas distribuições mostra a separação limitada.")

    # ---- desenvolvimento x teste
    secao("Desenvolvimento (validação cruzada) × teste", "modelo", "O AUC do desenvolvimento é otimista porque a seleção do modelo usa o mesmo treino.")
    cmp_ = pd.DataFrame([{"Modelo": nm(m["nome"]), "AUC na validação cruzada": m["cv"]["AUC média"], "± desvio": m["cv"]["AUC desvio"],
                          "ROC AUC no teste": m["teste"]["ROC AUC"], "IC 95% (teste)": f"{br(m['auc_ic95'][0], 2)} – {br(m['auc_ic95'][1], 2)}"} for m in M])
    tabela(cmp_, {"AUC na validação cruzada": "%.3f", "± desvio": "%.3f", "ROC AUC no teste": "%.3f"})
    if R.get("estabilidade"):
        est = pd.DataFrame(R["estabilidade"])
        with st.expander("Estabilidade: AUC em 20 divisões treino/teste diferentes"):
            est = est.rename(columns={"auc_media": "AUC médio", "auc_desvio": "Desvio", "auc_min": "Mínimo", "auc_max": "Máximo",
                                      "pr_auc_media": "PR AUC médio", "prevalencia_media": "Prevalência média"})
            tabela(est, {c: "%.3f" for c in est.columns[1:]})
            st.caption("Hiperparâmetros já escolhidos; serve só para medir a incerteza. AUC = 0,5 é o acaso.")
    else:
        st.caption("A análise de estabilidade (20 divisões) não está nesta exportação; ela está na Seção 11 do notebook.")

    # ---- explicação curta
    secao("Como ler os erros")
    t = ms["teste"]
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="caixa"><h4>Falso positivo</h4>O modelo alerta, mas o cliente não estava insatisfeito. '
                f'Custo: tempo de revisão. {nm(sel)}: <b>{t["FP"]}</b> no teste.</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="caixa"><h4>Falso negativo</h4>O modelo não alerta e o cliente estava insatisfeito: o risco passa despercebido. '
                f'{nm(sel)}: <b>{t["FN"]}</b> de {r_te} casos de risco do teste.</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="caixa"><h4>Desbalanceamento</h4>Só {pct(R["base"]["proporcao_risco"])} das reuniões são de risco. '
                f'Prever "tudo seguro" acerta {pct(1 - prev_te, 0)} no teste sem detectar nenhum caso; por isso accuracy sozinha engana '
                f'e usamos recall, balanced accuracy e PR AUC.</div>', unsafe_allow_html=True)


# ============================================================================== aba 4 — insights de negócio
def aba_insights(R):
    if not R or "descritivo" not in R:
        sem_resultados("Os insights de negócio")
        return
    D, B = R["descritivo"], R["base"]
    T, G = D["termos"], D["regras"]
    n_r, n_n = T["n_risco"], T["n_nao_risco"]
    st.markdown('<div class="sub">Leitura comercial dos resultados exploratórios do notebook. Cada bloco traz um selo: '
                '<span class="selo base">Observado na base</span> contagens e médias; <span class="selo assoc">Associação textual</span> '
                'termos mais frequentes em um grupo (descritivo, <b>não causal</b>); '
                '<span class="selo regra">Regras de palavras-chave</span> indicadores sem validação de acerto. '
                'Nenhum bloco desta aba é saída do classificador.</div>', unsafe_allow_html=True)

    # ---- termos mais relevantes
    secao("Termos mais relevantes do corpus", "assoc", f"TF-IDF médio nas {B['linhas_validas']} reuniões rotuladas (unigramas e bigramas).")
    top = pd.DataFrame(T["top_tfidf"][:15])
    c1, c2 = st.columns([3, 2])
    with c1:
        fig, ax = nova_fig(6.4, 4.4)
        ax.barh(top["termo"][::-1], top["tfidf_medio"][::-1], color=NAVY)
        ax.set_xlabel("TF-IDF médio"); ax.xaxis.grid(True, color="#E3E8EF")
        mostrar(fig)
    with c2:
        t = top.rename(columns={"termo": "Termo", "tfidf_medio": "TF-IDF médio", "pct_reunioes": "% das reuniões"})
        t["% das reuniões"] = 100 * t["% das reuniões"]
        tabela(t, {"TF-IDF médio": "%.4f", "% das reuniões": "%.0f"})
    st.caption("Os termos de maior peso são expressões genéricas de conversa: descrevem o corpus, não distinguem risco de não risco.")

    # ---- termos associados a risco x não risco
    secao("Termos associados a risco × não risco", "assoc",
          f"Diferença do TF-IDF médio entre reuniões com NPS &lt; 7 ({n_r}) e NPS ≥ 7 ({n_n}). Âmbar = mais frequente no grupo de risco; verde-azulado = no grupo sem risco.")
    ar = pd.DataFrame(T["associados_risco"][:10]); an = pd.DataFrame(T["associados_nao_risco"][:10])
    rotulos = list(an["termo"][::-1]) + list(ar["termo"][::-1])
    vals = list(-an["diferenca"][::-1]) + list(ar["diferenca"][::-1])
    fig, ax = nova_fig(9.5, 5.2)
    ax.barh(rotulos, vals, color=[TEAL] * len(an) + [AMBER] * len(ar))
    ax.axvline(0, color=NAVY, lw=1)
    ax.set_xlabel("diferença de TF-IDF médio (← sem risco | risco →)"); ax.xaxis.grid(True, color="#E3E8EF")
    mostrar(fig)
    ca, cb = st.columns(2)
    for col, df_, titulo in ((ca, T["associados_risco"], "Mais frequentes em reuniões de risco"), (cb, T["associados_nao_risco"], "Mais frequentes em reuniões sem risco")):
        t = pd.DataFrame(df_[:10]).rename(columns={"termo": "Termo", "tfidf_risco": "TF-IDF (risco)", "tfidf_nao_risco": "TF-IDF (sem risco)", "diferenca": "Diferença"})
        col.markdown(f"**{titulo}**")
        with col:
            tabela(t, {c: "%.4f" for c in t.columns[1:]})
    st.caption(f"Associação descritiva, não causal: as {n_r} reuniões de risco vêm de poucos clientes ({B.get('clientes_distintos', '—')} clientes no total) e os termos incluem "
               "nomes e assuntos específicos deles. Não são resultado do classificador nem provam poder preditivo.")

    # ---- produtos e categorias
    secao("Produtos e categorias mais mencionados", "regra", "Reuniões rotuladas em que o produto/categoria aparece (uma reunião pode citar vários).")
    men = pd.DataFrame(G["mencoes"])
    if len(men):
        men = men.sort_values("total")
        fig, ax = nova_fig(9.5, max(2.6, 0.55 * len(men) + 1.0))
        ax.barh(men["item"], men["total"], color=NAVY)
        for y, v in enumerate(men["total"]):
            ax.text(v + max(men["total"]) * 0.01, y, str(v), va="center", fontsize=9, color=NAVY)
        ax.set_xlabel("reuniões que mencionam"); ax.xaxis.grid(True, color="#E3E8EF")
        mostrar(fig)
        t = men.iloc[::-1].rename(columns={"item": "Produto / categoria", "total": "Reuniões", "risco": "Com NPS < 7"})
        t["% com NPS < 7"] = 100 * t["Com NPS < 7"] / t["Reuniões"]
        tabela(t[["Produto / categoria", "Reuniões", "Com NPS < 7", "% com NPS < 7"]], {"% com NPS < 7": "%.1f"})
        st.caption(f"Referência: {pct(B['proporcao_risco'])} de todas as reuniões rotuladas têm NPS < 7. Produtos sem menção explícita não aparecem.")
    else:
        st.write("Nenhuma menção identificada pelas regras.")

    # ---- temas, sentimento, oportunidade: tabela única comparando os grupos
    secao("Temas, sentimento e oportunidades (regras) por grupo de NPS", "regra",
          "Proporção de reuniões com o sinal em cada grupo. Se um indicador separasse bem os grupos, as duas últimas colunas seriam bem diferentes.")
    def linhas_grupo(titulo, itens, filtro=None):
        out = []
        for it in itens:
            if filtro and it["item"] not in filtro:
                continue
            out.append({"Indicador": titulo, "Sinal": it["item"], "Reuniões": it["total"],
                        f"% no grupo de risco (n={G['n_risco']})": 100 * it["risco"] / max(G["n_risco"], 1),
                        f"% no grupo sem risco (n={G['n_nao_risco']})": 100 * it["nao_risco"] / max(G["n_nao_risco"], 1)})
        return out
    linhas = (linhas_grupo("Sentimento", G["sentimento"], {"Positivo", "Negativo", "Misto"})
              + linhas_grupo("Oportunidade comercial", G["oportunidade"], {"Alta", "Média"})
              + linhas_grupo("Tema de problema", G["temas_problema"])
              + linhas_grupo("Sinal textual", G["sinais_risco_texto"]) + linhas_grupo("Sinal textual", G["duvida_comercial"]))
    if linhas:
        tg = pd.DataFrame(linhas)
        tabela(tg, {c: "%.1f" for c in tg.columns[3:]})
        st.caption("Sentimento, oportunidade e temas vêm de regras de palavras-chave (negação e problema já resolvido tratados), sem validação de acerto. "
                   "Os temas contam apenas menções com evidência de problema ativo (não resolvido).")

    # ---- NPS por segmento e faixa (resumo; detalhes na aba 2)
    secao("Onde o NPS médio é maior e menor", "base", "Resumo; gráficos e tabelas completos na aba Visão geral da base (grupos com ≥ 10 reuniões).")
    linhas = []
    for chave, rot in (("por_segmento", "Segmento"), ("por_faixa", "Faixa de faturamento")):
        g = pd.DataFrame(D[chave]); g = g[g["reunioes"] >= 10].sort_values("nps_medio")
        if len(g) >= 4:
            for pos_, sub in (("Menores", g.head(2)), ("Maiores", g.tail(2).iloc[::-1])):
                for _, r_ in sub.iterrows():
                    linhas.append({"Dimensão": rot, "Posição": pos_, "Grupo": nome_segmento(r_["grupo"]) if chave == "por_segmento" else str(r_["grupo"]), "Reuniões": int(r_["reunioes"]), "NPS médio": r_["nps_medio"]})
    if linhas:
        tabela(pd.DataFrame(linhas), {"NPS médio": "%.2f"})
        st.caption(f"Média geral: {br(D['nps_media'], 2)}. Diferenças descritivas entre grupos pequenos podem ser acaso; não indicam causa.")


# ============================================================================== aba 5 — como funciona
def aba_como_funciona(R, art):
    B = (R or {}).get("base") or (art or {}).get("relatorio_base") or {}
    D = (R or {}).get("descritivo") or {}
    st.markdown('<div class="sub">Da transcrição à recomendação: o que cada etapa faz e o que é (ou não é) um modelo treinado.</div>', unsafe_allow_html=True)

    # ---- fluxo
    passos = [("", "Transcrição", "Texto completo de uma reunião, anonimizado."),
              ("", "Limpeza textual", "Minúsculas, sem marcações e stopwords; negações mantidas."),
              ("", "TF-IDF", "Converte o texto em números: o peso de cada termo."),
              ("m", "Classificação experimental", "Regressão Logística: score e previsão de risco."),
              ("r", "Indicadores auxiliares", "Produto, sentimento, oportunidade e temas."),
              ("", "Recomendações", "Ações com o motivo; revisão humana se houver conflito.")]
    tags = {"m": '<div class="fx-g">Modelo</div>', "r": '<div class="fx-g">Regras</div>'}
    html_passos = "".join(f'<div class="fx-s {c}"><div class="fx-c">{i}</div>{tags.get(c, "")}<div class="fx-n">{n}</div><div class="fx-d">{d}</div></div>'
                          for i, (c, n, d) in enumerate(passos, 1))
    st.markdown(f'<div class="fx"><div class="fx-t">Fluxo da solução</div><div class="fx-row">{html_passos}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="leg"><span><i style="background:#fff;border:1px solid #999"></i>Processamento</span>'
                f'<span><i style="background:{TEAL}"></i>Classificador treinado (supervisionado)</span>'
                '<span><i style="border:2px dashed #8493A5;width:7px;height:7px"></i>Regras de palavras-chave (não treinadas)</span></div>', unsafe_allow_html=True)

    # ---- conceitos
    secao("Conceitos-chave")
    risco, n = B.get("risco"), B.get("linhas_validas")
    prop = (risco / n) if risco is not None and n else 0.158
    limiar_score = None
    if art:
        try:
            limiar_score = score_percentil(art, art["limiar"])
        except Exception:
            limiar_score = None
    L = limiar_score if limiar_score is not None else 50
    base_txt = f"{risco} de {n} reuniões rotuladas ({pct(prop)})" if risco is not None and n else "uma minoria das reuniões rotuladas"
    c1 = ('<div class="kc"><div class="k">Variável-alvo</div><h4>Como o risco é definido</h4>'
          '<div class="pills"><span class="pill r">NPS &lt; 7 → 1 · risco</span><span class="pill s">NPS ≥ 7 → 0 · sem risco</span></div>'
          f'<div class="prop"><div style="width:{100 * prop:.1f}%"></div></div><div class="cap">Risco: {base_txt}</div>'
          '<p>É insatisfação medida pelo NPS, <b>não</b> cancelamento (churn). Reuniões sem NPS não recebem rótulo.</p></div>')
    c2 = ('<div class="kc"><div class="k">Representação do texto</div><h4>Para que serve o TF-IDF</h4>'
          '<div class="chips"><span class="chip alto">trava</span><span class="chip alto">cancelar</span><span class="chip baixo">bom dia</span><span class="chip baixo">tudo bem</span></div>'
          '<div class="cap">Ilustração conceitual (sem valores reais): termos específicos pesam mais; os comuns, menos.</div>'
          '<p>Dá mais peso ao que é típico da reunião e raro no conjunto. O classificador aprende com esses pesos, não com o sentido das frases.</p></div>')
    c3 = ('<div class="kc"><div class="k">Leitura da saída</div><h4>Score e limiar</h4>'
          f'<div class="gauge" style="background:linear-gradient(90deg,{TEAL} 0%,{TEAL} {L:.0f}%,{AMBER} {L:.0f}%,{AMBER} 100%)">'
          f'<div class="mk" style="left:{L:.0f}%">limiar {L:.0f}</div></div><div class="esc"><span>0 · seguro</span><span>risco · 100</span></div>'
          '<p>O <b>score</b> é um percentil (0–100), ordinal e <b>não calibrado</b>. O <b>limiar</b> foi definido só no treino (maior recall com ≥ 50% de especificidade); '
          'mudá-lo altera quantos alertas saem, não a qualidade do modelo.</p></div>')
    st.markdown(f'<div class="kc-grid">{c1}{c2}{c3}</div>', unsafe_allow_html=True)

    # ---- classificador x regras
    secao("Classificador × regras de palavras-chave")
    def linhas(itens):
        return "".join(f"<div class='rw'><b>{k}</b><span>{v}</span></div>" for k, v in itens)
    esq = ('<div class="vs"><div class="k">Supervisionado</div><h4>Classificador</h4>' + linhas([
        ("Aprende com", "reuniões com NPS informado"), ("Entrega", "previsão (risco, seguro ou inconclusivo) e score"),
        ("Medido por", "ROC AUC, PR AUC, recall… todas fracas no teste"), ("No app", "previsão experimental")]) + "</div>")
    dir_ = ('<div class="vs reg"><div class="k">Não treinadas</div><h4>Regras de palavras-chave</h4>' + linhas([
        ("Aprendem com", "nada: listas de termos, com negação e problema já resolvido"), ("Entregam", "produto, sentimento, oportunidade, temas e termos de risco"),
        ("Medidas por", "sem métrica: a base não tem rótulos para isso"), ("No app", "indicadores auxiliares")]) + "</div>")
    st.markdown(f'<div class="vs-grid">{esq}<div class="vs-x">×</div>{dir_}</div>'
                '<div class="nt">Quando os dois divergem, o painel mostra o conflito e recomenda revisão humana.</div>', unsafe_allow_html=True)

    # ---- limites e próximos passos
    secao("Limites e próximos passos")
    v = D.get("nps_vs_reuniao")
    n_cli, n_cli_r = B.get("clientes_distintos", "poucos"), D.get("clientes_risco", "poucos")
    lim = [("Poucos exemplos de risco", f"{n_cli} clientes, dos quais {n_cli_r} com risco."),
           ("Rótulo distante da reunião", (f"{pct(v['pct_anterior'], 0)} dos NPS são anteriores à reunião (mediana de {br(abs(v['mediana_dias']), 0)} dias)." if v else "o NPS costuma ser anterior à reunião.")),
           ("Teste pequeno e sem ganho", "poucos casos de risco; nenhum modelo superou o acaso."),
           ("Textos curtos", "ficam “Inconclusivos”: o modelo precisa de transcrições completas.")]
    ok = [("Rótulo próximo da reunião", "satisfação coletada depois dela e, idealmente, o churn real."),
          ("Mais clientes e validação temporal", "treinar no passado, testar no futuro e por cliente."),
          ("Calibração e custos", "probabilidades calibradas; custo de falso positivo × negativo com a área comercial."),
          ("Regras validadas", "anotação humana e piloto com revisão humana antes de qualquer decisão automática.")]
    def lista(itens, css, icone):
        return f'<ul class="il {css}">' + "".join(f'<li><span class="ic">{icone}</span><span><b>{t}</b>{d}</span></li>' for t, d in itens) + "</ul>"
    st.markdown(f'<div class="duo"><div class="kc lim-c"><div class="k">Atenção</div><h4>Por que a qualidade ainda é limitada</h4>{lista(lim, "lim", "!")}</div>'
                f'<div class="kc ok-c"><div class="k">Caminho</div><h4>O que seria necessário para uso confiável</h4>{lista(ok, "ok", "✓")}</div></div>', unsafe_allow_html=True)


# ============================================================================== página
st.markdown('<div class="hero"><h1>Painel de Inteligência Comercial</h1>'
            '<p>Protótipo acadêmico · FIAP Challenge 2026 · TOTVS — apoio à leitura de transcrições de reuniões, '
            'não substitui a análise do consultor.</p></div>', unsafe_allow_html=True)

art = carregar_modelo()
R = carregar_resultados()
if art is None:
    st.error("Modelo não encontrado (modelo_risco.joblib). Rode `python pipeline.py treinar`. "
             "Sem ele, só os indicadores por regras são exibidos.")
else:
    rel = art.get("relatorio_base", {})
    if art.get("base_sintetica"):
        aviso = "Modelo treinado em base <b>sintética</b> (demonstração técnica): não representa clientes reais."
    else:
        aviso = (f"<b>Modelo experimental.</b> Treinado com {rel.get('linhas_validas', '?')} reuniões originais "
                 f"({rel.get('risco', '?')} de risco, NPS &lt; 7) e <b>sem desempenho melhor que o acaso</b> no teste "
                 f"(ROC AUC {br(art['metricas_teste']['ROC AUC'], 2)}). Estima insatisfação medida pelo NPS, "
                 "<b>não</b> prevê cancelamento. Use os resultados como apoio, nunca como decisão automática.")
    st.markdown(f'<div class="aviso">{aviso}</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Analisar transcrição", "Visão geral da base", "Desempenho dos modelos",
                                        "Insights de negócio", "Como funciona"])
with tab1:
    aba_analisar(art)
with tab2:
    aba_base(R)
with tab3:
    aba_modelos(R, art)
with tab4:
    aba_insights(R)
with tab5:
    aba_como_funciona(R, art)
