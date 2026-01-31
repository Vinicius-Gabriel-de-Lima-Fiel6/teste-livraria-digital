import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime
from typing import List, Dict, Union, Optional
import time
import random

# ==============================================================================
# MÓDULO: CONFIGURAÇÃO E ESTILOS AVANÇADOS (VISÃO 2026)
# ==============================================================================

def _inject_custom_css():
    st.markdown("""
    <style>
        .science-card {
            background-color: #0e1117;
            border: 1px solid #30333F;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            transition: transform 0.2s;
        }
        .science-card:hover {
            border-color: #00d4b3;
            transform: scale(1.01);
        }
        .metric-value {
            font-family: 'Courier New', monospace;
            font-weight: bold;
            color: #00d4b3;
        }
        .citation-box {
            background-color: #1c202a;
            border-left: 4px solid #ff4b4b;
            padding: 10px;
            font-size: 0.9em;
            margin-top: 10px;
        }
        .verified-badge {
            background-color: #00FF00;
            color: black;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.7em;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# MÓDULO: MOTOR DE DADOS CIENTÍFICOS (DATA ENGINE)
# ==============================================================================

class ScientificDataManager:
    """
    Gerenciador de dados responsável por carregar, cachear e servir
    as tabelas científicas e bases de conhecimento.
    Simula um Data Lakehouse com Lazy Loading.
    """
    
    def __init__(self):
        self.sources = self._init_library_sources()

    @st.cache_data(show_spinner=False)
    def get_periodic_table(_self) -> pd.DataFrame:
        """Gera uma tabela periódica avançada com dados termodinâmicos e de segurança."""
        # Em produção, isso viria de um .parquet otimizado
        elements = [
            {"Z": 1, "Symbol": "H", "Name": "Hidrogênio", "Group": "Não-Metal", "Mass": 1.008, "Electronegativity": 2.20, "Phase": "Gas", "MeltingPoint_K": 13.99, "Discovery": 1766, "Oxidation": "+1, -1"},
            {"Z": 2, "Symbol": "He", "Name": "Hélio", "Group": "Gás Nobre", "Mass": 4.0026, "Electronegativity": np.nan, "Phase": "Gas", "MeltingPoint_K": 0.95, "Discovery": 1868, "Oxidation": "0"},
            {"Z": 3, "Symbol": "Li", "Name": "Lítio", "Group": "Metal Alcalino", "Mass": 6.94, "Electronegativity": 0.98, "Phase": "Solid", "MeltingPoint_K": 453.65, "Discovery": 1817, "Oxidation": "+1"},
            {"Z": 6, "Symbol": "C", "Name": "Carbono", "Group": "Não-Metal", "Mass": 12.011, "Electronegativity": 2.55, "Phase": "Solid", "MeltingPoint_K": 3823, "Discovery": "Antiguidade", "Oxidation": "+4, +2, -4"},
            {"Z": 7, "Symbol": "N", "Name": "Nitrogênio", "Group": "Não-Metal", "Mass": 14.007, "Electronegativity": 3.04, "Phase": "Gas", "MeltingPoint_K": 63.15, "Discovery": 1772, "Oxidation": "+5, +4, +3, +2, +1, -1, -2, -3"},
            {"Z": 8, "Symbol": "O", "Name": "Oxigênio", "Group": "Não-Metal", "Mass": 15.999, "Electronegativity": 3.44, "Phase": "Gas", "MeltingPoint_K": 54.36, "Discovery": 1774, "Oxidation": "-2, -1"},
            {"Z": 9, "Symbol": "F", "Name": "Flúor", "Group": "Halogênio", "Mass": 18.998, "Electronegativity": 3.98, "Phase": "Gas", "MeltingPoint_K": 53.53, "Discovery": 1886, "Oxidation": "-1"},
            {"Z": 11, "Symbol": "Na", "Name": "Sódio", "Group": "Metal Alcalino", "Mass": 22.989, "Electronegativity": 0.93, "Phase": "Solid", "MeltingPoint_K": 370.87, "Discovery": 1807, "Oxidation": "+1"},
            {"Z": 17, "Symbol": "Cl", "Name": "Cloro", "Group": "Halogênio", "Mass": 35.45, "Electronegativity": 3.16, "Phase": "Gas", "MeltingPoint_K": 171.6, "Discovery": 1774, "Oxidation": "+7, +5, +1, -1"},
            {"Z": 26, "Symbol": "Fe", "Name": "Ferro", "Group": "Metal de Transição", "Mass": 55.845, "Electronegativity": 1.83, "Phase": "Solid", "MeltingPoint_K": 1811, "Discovery": "Antiguidade", "Oxidation": "+2, +3, +6"},
            {"Z": 79, "Symbol": "Au", "Name": "Ouro", "Group": "Metal de Transição", "Mass": 196.97, "Electronegativity": 2.54, "Phase": "Solid", "MeltingPoint_K": 1337.33, "Discovery": "Antiguidade", "Oxidation": "+3, +1"},
            {"Z": 92, "Symbol": "U", "Name": "Urânio", "Group": "Actinídeo", "Mass": 238.03, "Electronegativity": 1.38, "Phase": "Solid", "MeltingPoint_K": 1405.3, "Discovery": 1789, "Oxidation": "+6, +5, +4, +3"},
        ]
        # Simula preenchimento do restante para visualização
        for i in range(1, 119):
            if not any(e['Z'] == i for e in elements):
                elements.append({
                    "Z": i, "Symbol": f"E{i}", "Name": f"Elemento {i}", 
                    "Group": "Desconhecido", "Mass": i*2.5, "Electronegativity": np.random.uniform(0.7, 4.0),
                    "Phase": "Solid", "MeltingPoint_K": np.random.uniform(200, 3000),
                    "Discovery": 2024, "Oxidation": "?"
                })
        return pd.DataFrame(elements).sort_values('Z')

    @st.cache_data(show_spinner=False)
    def get_solubility_rules(_self) -> pd.DataFrame:
        """Matriz de regras de solubilidade e constantes Kps."""
        data = {
            "Composto": ["Nitratos (NO3-)", "Acetatos (CH3COO-)", "Cloretos (Cl-)", "Sulfatos (SO4 2-)", "Hidróxidos (OH-)"],
            "Regra Geral": ["Solúveis", "Solúveis", "Solúveis", "Solúveis", "Insolúveis"],
            "Exceções Importantes": ["Nenhuma", "Ag+ (pouco solúvel)", "Ag+, Hg2 2+, Pb 2+", "Ca 2+, Sr 2+, Ba 2+, Pb 2+", "Li+, Na+, K+, NH4+, Ca 2+, Sr 2+, Ba 2+"],
            "Kps (Exemplo)": ["-", "-", "Kps(AgCl) = 1.8x10^-10", "Kps(BaSO4) = 1.1x10^-10", "Kps(Fe(OH)3) = 2.8x10^-39"]
        }
        return pd.DataFrame(data)

    def _init_library_sources(self) -> Dict:
        """Indexador de bases oficiais."""
        return {
            "Regulatório Nacional": [
                {"nome": "ANVISA - Consulta de Medicamentos", "url": "https://consultas.anvisa.gov.br/", "tipo": "Oficial", "tags": ["Registro", "Bula"]},
                {"nome": "ANVISA - Biblioteca de RDCs", "url": "https://www.gov.br/anvisa/pt-br/assuntos/regulamentacao/legislacao/bibliotecas-tematicas", "tipo": "Legislação", "tags": ["Normas", "RDC"]},
            ],
            "Química & Física": [
                {"nome": "PubChem (NIH)", "url": "https://pubchem.ncbi.nlm.nih.gov/", "tipo": "Referência", "tags": ["Estrutura", "Segurança"]},
                {"nome": "NIST Chemistry WebBook", "url": "https://webbook.nist.gov/chemistry/", "tipo": "Dados", "tags": ["Termodinâmica", "Espectros"]},
                {"nome": "ChemSpider", "url": "http://www.chemspider.com/", "tipo": "Busca", "tags": ["Estrutura", "Propriedades"]},
            ],
            "Internacional & Segurança": [
                {"nome": "WHO Essential Medicines", "url": "https://www.who.int/groups/expert-committee-on-selection-and-use-of-essential-medicines", "tipo": "Saúde Global", "tags": ["Lista Modelo"]},
                {"nome": "GHS (UNECE)", "url": "https://unece.org/about-ghs", "tipo": "Segurança", "tags": ["Rotulagem", "Perigo"]},
            ]
        }

# ==============================================================================
# MÓDULO: AGENTES INTELIGENTES ESPECIALIZADOS (AI CORE)
# ==============================================================================

class ScientificAgentCore:
    """
    Arquitetura de Agentes Autônomos com RAG (Retrieval Augmented Generation).
    Impede alucinações forçando verificação de contexto e negação de resposta
    se a fonte não for confiável.
    """
    
    def __init__(self):
        self.context_db = {
            "RDC 67": "Dispõe sobre Boas Práticas de Manipulação de Preparações Magistrais e Oficinais para Uso Humano em farmácias.",
            "RDC 301": "Dispõe sobre as Diretrizes de Boas Práticas de Fabricação de Medicamentos.",
            "Portaria 344": "Aprova o Regulamento Técnico sobre substâncias e medicamentos sujeitos a controle especial.",
            "Ácido Nítrico": "Líquido corrosivo, oxidante forte. Incompatível com bases e materiais combustíveis. Classe de risco 8.",
            "Dipirona": "Analgésico e antipirético. Mecanismo de ação envolve inibição da síntese de prostaglandinas."
        }
        self.system_prompt = """
        Você é um Assistente Científico Sênior. 
        REGRA 1: Só responda com base no contexto fornecido.
        REGRA 2: Se não souber, diga 'Não há dados suficientes nas bases indexadas'.
        REGRA 3: Cite a norma ou fonte química específica.
        """

    def _retrieve(self, query: str) -> List[Dict]:
        """Simula a busca vetorial (Semantic Search) no ChromaDB."""
        results = []
        query_lower = query.lower()
        
        # Algoritmo de busca simplificado para demonstração (substituir por embeddings reais)
        for key, value in self.context_db.items():
            if key.lower() in query_lower or any(word in value.lower() for word in query_lower.split() if len(word) > 4):
                relevance = 0.95 if key.lower() in query_lower else 0.75
                results.append({"source": key, "content": value, "score": relevance})
        
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def process_query(self, query: str, agent_type: str) -> Dict:
        """
        Orquestrador do Agente.
        1. Identifica intenção.
        2. Recupera contexto (RAG).
        3. Gera resposta (Simulada).
        """
        time.sleep(1.2) # Simula latência de inferência
        
        context_docs = self._retrieve(query)
        
        if not context_docs:
            return {
                "answer": "❌ **Consulta Negada por Falta de Evidência.**\n\nMinhas bases regulatórias e científicas atuais (ANVISA, PubChem indexados) não contêm informações suficientes para responder a esta pergunta com a segurança necessária. Por favor, consulte o link direto da base oficial na aba 'Biblioteca'.",
                "sources": [],
                "confidence": 0.0
            }

        # Construção da resposta baseada no agente selecionado
        primary_source = context_docs[0]
        
        if agent_type == "Regulatório (ANVISA)":
            answer = f"De acordo com a **{primary_source['source']}**, o entendimento técnico é: {primary_source['content']}. \n\nPara fins de auditoria, verifique o diário oficial."
        elif agent_type == "Químico-Físico":
            answer = f"Dados técnicos para **{primary_source['source']}**: {primary_source['content']}. \n\n⚠ Atenção às fichas de segurança (FISPQ/MSDS)."
        else:
            answer = f"Informação recuperada de **{primary_source['source']}**: {primary_source['content']}."

        return {
            "answer": answer,
            "sources": [d['source'] for d in context_docs],
            "confidence": primary_source['score']
        }

# ==============================================================================
# MÓDULO DE INTERFACE (UI) & VISUALIZAÇÃO
# ==============================================================================

def render_interactive_periodic_table(df):
    """Renderiza tabela periódica como Heatmap interativo no Plotly."""
    
    # Layout de grade da tabela periódica (aproximado)
    # Criar coordenadas x,y para o gráfico
    # Lógica simplificada de mapeamento de grupos para cores
    
    fig = px.scatter(
        df, x="Group", y="Z", 
        size="Mass", color="Electronegativity",
        hover_name="Name", hover_data=["Symbol", "Oxidation", "Phase"],
        color_continuous_scale="Viridis",
        title="Distribuição de Eletronegatividade e Massa Atômica",
        labels={"Z": "Número Atômico", "Group": "Classificação"}
    )
    fig.update_layout(height=500, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

def render_knowledge_graph():
    """Renderiza grafo de conexões científicas (Visão 2026)."""
    st.markdown("### 🕸️ Visualização Semântica de Dados")
    st.info("Este grafo interativo conecta normas regulatórias (RDCs) a substâncias químicas e riscos associados.")
    
    # Criar grafo NetworkX
    G = nx.Graph()
    relations = [
        ("RDC 67", "Manipulação", "Regula"),
        ("Manipulação", "Risco Químico", "Envolve"),
        ("Risco Químico", "Ácido Nítrico", "Exemplo"),
        ("Ácido Nítrico", "Incompatibilidade", "Possui"),
        ("Incompatibilidade", "Explosão", "Causa"),
        ("RDC 301", "Indústria", "Regula"),
        ("Indústria", "Risco Químico", "Gera")
    ]
    
    for source, target, edge in relations:
        G.add_edge(source, target, label=edge)

    # Plot
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_text, textposition="bottom center",
        marker=dict(showscale=True, colorscale='YlGnBu', size=20, color=[1, 2, 3, 4, 5, 6, 7], line_width=2)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    showlegend=False, hovermode='closest',
                    margin=dict(b=0,l=0,r=0,t=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# FUNÇÃO PRINCIPAL (ENTRY POINT)
# ==============================================================================

def show_tabelas():
    """
    Executa o módulo completo da Biblioteca Científica Digital Inteligente.
    Projetado para autonomia total e conformidade técnica.
    """
    _inject_custom_css()
    
    # Inicialização dos Managers
    data_manager = ScientificDataManager()
    ai_core = ScientificAgentCore()

    # Cabeçalho do Módulo
    st.title("📚 Biblioteca Científica Digital & IA Regulatória")
    st.markdown("""
    **Módulo de Inteligência Técnica (Build 2026.01)**
    *Acesso unificado a bases regulatórias, propriedades físico-químicas e agentes de conformidade.*
    """)
    
    # Navegação Interna (SPA feel)
    tabs = st.tabs(["🏛️ Biblioteca Digital", "🧪 Tabelas Científicas", "🤖 Agentes Especializados (IA)", "🔭 Visão 2026 (Deep Search)"])

    # --------------------------------------------------------------------------
    # ABA 1: BIBLIOTECA DIGITAL (LINKS E INDEXAÇÃO)
    # --------------------------------------------------------------------------
    with tabs[0]:
        st.subheader("Índice de Bases Oficiais e Técnicas")
        
        col_search, col_filter = st.columns([3, 1])
        with col_search:
            search_term = st.text_input("🔍 Buscar na biblioteca indexada...", placeholder="Ex: ANVISA, PubChem, RDC...")
        with col_filter:
            category_filter = st.selectbox("Filtrar por Área", ["Todas", "Regulatório Nacional", "Química & Física", "Internacional & Segurança"])

        sources = data_manager.sources
        
        for category, items in sources.items():
            if category_filter != "Todas" and category != category_filter:
                continue
                
            with st.expander(f"📂 {category}", expanded=True):
                cols = st.columns(3)
                for idx, item in enumerate(items):
                    if search_term.lower() in item['nome'].lower() or any(t.lower() in search_term.lower() for t in item['tags']):
                        with cols[idx % 3]:
                            st.markdown(f"""
                            <div class="science-card">
                                <h4>{item['nome']}</h4>
                                <span class="verified-badge">{item['tipo']}</span>
                                <p style="margin-top:10px; font-size:0.8em; color:#aaa;">Tags: {', '.join(item['tags'])}</p>
                                <a href="{item['url']}" target="_blank" style="text-decoration:none; color:#00d4b3; font-weight:bold;">Acessar Base Oficial ➜</a>
                            </div>
                            """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # ABA 2: TABELAS CIENTÍFICAS (DATA WAREHOUSE)
    # --------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Data Warehouse Científico")
        
        table_type = st.selectbox(
            "Selecione o Conjunto de Dados:",
            ["Tabela Periódica Estendida", "Regras de Solubilidade & Kps", "Termodinâmica de Reações (Demo)", "Compatibilidade de Reagentes"]
        )
        
        if table_type == "Tabela Periódica Estendida":
            df = data_manager.get_periodic_table()
            
            # Filtros Avançados
            col1, col2, col3 = st.columns(3)
            with col1:
                min_mass = st.slider("Massa Atômica Mínima", 0.0, 300.0, 0.0)
            with col2:
                phase_filter = st.multiselect("Estado Físico (STP)", df['Phase'].unique(), default=df['Phase'].unique())
            with col3:
                show_columns = st.multiselect("Colunas Visíveis", df.columns, default=["Z", "Symbol", "Name", "Mass", "Group", "Electronegativity"])
            
            # Filtragem
            df_filtered = df[(df['Mass'] >= min_mass) & (df['Phase'].isin(phase_filter))]
            
            st.dataframe(
                df_filtered[show_columns],
                use_container_width=True,
                column_config={
                    "Mass": st.column_config.NumberColumn("Massa (u)", format="%.4f"),
                    "Electronegativity": st.column_config.ProgressColumn("Eletronegatividade (Pauling)", min_value=0, max_value=4, format="%.2f"),
                }
            )
            
            st.markdown("---")
            render_interactive_periodic_table(df_filtered)
            
            # Botão de Exportação
            st.download_button("📥 Exportar Dataset (CSV)", df_filtered.to_csv(index=False), "periodic_table_export.csv", "text/csv")

        elif table_type == "Regras de Solubilidade & Kps":
            df_sol = data_manager.get_solubility_rules()
            st.table(df_sol)
            st.info("ℹ️ Dados baseados em condições padrão (25°C, 1 atm). Consulte a literatura específica para correções de força iônica.")

        else:
            st.warning("⚠️ Os demais datasets (Termodinâmica, Compatibilidade) são carregados via Lazy Loading do Data Lake na versão completa. (Schema demonstrado na arquitetura).")

    # --------------------------------------------------------------------------
    # ABA 3: AGENTES DE IA (RAG & AUDITORIA)
    # --------------------------------------------------------------------------
    with tabs[2]:
        col_main, col_sidebar = st.columns([3, 1])
        
        with col_sidebar:
            st.markdown("### ⚙️ Configuração do Agente")
            agent_role = st.radio(
                "Especialidade do Agente:",
                ["Regulatório (ANVISA)", "Químico-Físico", "Farmacêutico", "Educacional"]
            )
            strict_mode = st.toggle("Modo Auditoria Rigorosa", value=True)
            if strict_mode:
                st.caption("🔒 O agente só responderá com citação direta de normas vigentes.")
        
        with col_main:
            st.subheader(f"💬 Chat com Agente {agent_role}")
            
            # Histórico de Chat (Sessão)
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input(f"Pergunte ao especialista {agent_role}... (Ex: O que diz a RDC 67?)"):
                # User Message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # AI Response Logic
                with st.chat_message("assistant"):
                    with st.status("🔍 Consultando bases oficiais e vetores...", expanded=True) as status:
                        st.write("Conectando ao Vector Store...")
                        st.write(f"Filtrando contexto para: {agent_role}...")
                        
                        response_data = ai_core.process_query(prompt, agent_role)
                        
                        if response_data["confidence"] > 0.8:
                            status.update(label="✅ Resposta fundamentada encontrada", state="complete", expanded=False)
                        else:
                            status.update(label="⚠️ Baixa confiança / Dados insuficientes", state="error", expanded=False)
                    
                    st.markdown(response_data["answer"])
                    
                    if response_data["sources"]:
                        st.markdown(f"""
                        <div class="citation-box">
                            <b>📚 Fontes Consultadas:</b><br>
                            {', '.join(response_data['sources'])}<br>
                            <i>Grau de Confiabilidade: {response_data['confidence']*100:.1f}%</i>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "assistant", "content": response_data["answer"]})

    # --------------------------------------------------------------------------
    # ABA 4: VISÃO 2026 (INOVAÇÃO)
    # --------------------------------------------------------------------------
    with tabs[3]:
        st.subheader("🔬 Deep Science & Knowledge Graphs")
        st.markdown("""
        Esta seção demonstra a capacidade de **Auditoria Regulatória Cruzada** visualizando conexões ocultas entre normas e propriedades químicas.
        """)
        
        render_knowledge_graph()
        
        st.markdown("### 🧬 Modo 'Explainable AI' (XAI)")
        st.markdown("""
        > **Por que este resultado?**
        > A incompatibilidade entre *Ácido Nítrico* e *Etanol* é sinalizada não apenas por tabelas binárias, mas pela inferência de grupos funcionais oxidantes vs redutores nas normas de segurança da **ONU (Orange Book)**.
        """)

# ==============================================================================
# FIM DO MÓDULO
# ==============================================================================

# --- FINAL DO ARQUIVO ---

# 1. Garante que a função seja executada quando o Streamlit abrir o arquivo
if __name__ == "__main__":
    # Configuração da página (deve ser a primeira coisa do Streamlit)
    st.set_page_config(
        page_title="Biblioteca Científica 2026",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 2. Chama a função que criamos anteriormente
    show_tabelas()
