from dotenv import load_dotenv
load_dotenv()

import os
from flask import Flask, render_template, request
from markupsafe import Markup
from crewai import Agent, Task, Crew, Process


from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from crewai.tools import tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import markdown

from langchain_community.llms import Ollama

app = Flask(__name__)

ollama_llm = Ollama(model="llama3", base_url="http://localhost:11434")


def setup_knowledge_base_from_pdfs():
    all_chunks = []
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    knowledge_dir = 'knowledge_base'
    if not os.path.exists(knowledge_dir): return []
    for filename in os.listdir(knowledge_dir):
        if filename.endswith('.pdf'):
            try:
                reader = PdfReader(os.path.join(knowledge_dir, filename))
                raw_text = ''
                for page in reader.pages:
                    text = page.extract_text()
                    if text: raw_text += text
                chunks = text_splitter.split_text(raw_text)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Erro ao processar o arquivo {filename}: {e}")
    return all_chunks

knowledge_base_chunks = setup_knowledge_base_from_pdfs()
if knowledge_base_chunks:
    vectorizer = TfidfVectorizer().fit(knowledge_base_chunks)
    knowledge_vectors = vectorizer.transform(knowledge_base_chunks)
else:
    print("AVISO: Nenhuma base de conhecimento carregada.")
    vectorizer = None
    knowledge_vectors = None

@tool("Ferramenta de Busca em Documentos PDF")
def rag_tool(query: str) -> str:
    """Busca informações relevantes na base de conhecimento em PDF da empresa."""
    if knowledge_vectors is None:
        return "A base de conhecimento não está disponível."
    
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, knowledge_vectors).flatten()
    most_similar_index = np.argmax(similarities)
    return knowledge_base_chunks[most_similar_index]

def run_rag_crew(pergunta: str):
    retrieval_agent = Agent(
        role='Especialista em Pesquisa de Documentos',
        goal='Encontrar as informações mais relevantes em nossa base de conhecimento para responder à pergunta do usuário.',
        backstory='Você é um mestre em encontrar agulhas no palheiro.',
        tools=[rag_tool],
        verbose=False 
    )
    writer_agent = Agent(
        role='Redator Técnico',
        goal='Gerar uma resposta clara e concisa para o usuário, baseando-se estritamente no contexto fornecido.',
        backstory="""Você é o Dr. Klaus, um documentador técnico sênior com 20 anos de experiência na Siemens.
                        Sua especialidade é pegar informações brutas e densas de engenheiros e transformá-las em documentação clara, precisa e fácil de seguir.
                      Você odeia ambiguidade e sempre busca a clareza máxima. Sua obsessão é a precisão factual, baseando-se estritamente nas fontes fornecidas.""",
        verbose=False
    )
    
    retrieval_task = Task(description=f'Use a ferramenta de busca para encontrar o trecho relevante para a pergunta: "{pergunta}"', expected_output='O trecho (chunk) de texto mais relevante encontrado na base de conhecimento.', agent=retrieval_agent)
    writing_task = Task(description=f'Analise o contexto fornecido e responda à pergunta do usuário: "{pergunta}". Use SOMENTE as informações do contexto.', expected_output='Uma resposta completa e bem formatada em Markdown. A resposta deve conter:- Um título principal. - Um parágrafo introdutório que resume a resposta. - Se aplicável, use uma lista com marcadores (bullets) para detalhar os pontos importantes. - Use **negrito** para destacar termos-chave. - Se a resposta não estiver no contexto, responda educadamente: Não encontrei informações sobre isso nos documentos fornecidos.', context=[retrieval_task], agent=writer_agent)
    

    revisor_agent = Agent(
        role='Editor Técnico Chefe',
        goal='Revisar a documentação escrita para garantir precisão, clareza e conformidade com o estilo.',
        backstory='Você é o editor final. Seu trabalho é pegar um rascunho e garantir que ele esteja perfeito, verificando se a resposta corresponde EXATAMENTE ao contexto fornecido e se não há nenhuma informação inventada.',
        verbose=False,
        llm=ollama_llm
    )


    review_task = Task(
        description="""Leia o rascunho da resposta e o contexto original. Verifique os seguintes pontos:
        1. A resposta é 100% baseada no contexto?
        2. A resposta é clara e fácil de entender?
        3. A formatação está correta?
        Retorne a versão final e aprovada do texto.""",
        expected_output="O texto final em Markdown, revisado e aprovado.",
        context=[writing_task],
        agent=revisor_agent
    )

    crew = Crew(
        agents=[retrieval_agent, writer_agent, revisor_agent],
        tasks=[retrieval_task, writing_task, review_task],
        process=Process.sequential,
        llm=ollama_llm 
    )
    result = crew.kickoff()
    return result.raw

def run_procedure_crew(topic: str):
    style_expert_agent = Agent(
      role='Especialista em Documentação e Procedimentos',
      goal='Criar novos procedimentos operacionais que sigam estritamente o tom, formato e estilo dos documentos existentes.',
      backstory="""Você é um escritor técnico meticuloso com um olhar apurado para consistência.""",
      tools=[rag_tool],
      verbose=False
    )

    style_retrieval_task = Task(description=f"""Busque na base de conhecimento por exemplos de procedimentos ou seções que demonstrem o estilo de escrita padrão. O tópico do novo procedimento é '{topic}'. Sua busca deve focar em encontrar o FORMATO, TOM e ESTRUTURA.""", expected_output="""Um ou dois trechos de texto que sejam bons exemplos do estilo de escrita.""", agent=style_expert_agent)
    procedure_writing_task = Task(description=f"""Usando os *exemplos de estilo* fornecidos no contexto, escreva um novo procedimento passo a passo sobre '{topic}'. Replique o formato, o tom e a terminologia dos exemplos.""", expected_output=f"Um documento em formato Markdown contendo o procedimento completo para '{topic}'.", context=[style_retrieval_task], agent=style_expert_agent)
    
    crew = Crew(
        agents=[style_expert_agent],
        tasks=[style_retrieval_task, procedure_writing_task],
        process=Process.sequential,
        llm=ollama_llm 
    )

    result = crew.kickoff()
    return result.raw

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_answer', methods=['POST'])
def get_answer():
    pergunta = request.form['pergunta']
    if not pergunta:
        return render_template('index.html', error="Por favor, digite uma pergunta.")
    if not knowledge_base_chunks:
         return render_template('result.html', title="Erro", pergunta=pergunta, resultado="<p><b>Erro:</b> Nenhum documento PDF encontrado na pasta 'knowledge_base'.</p>")
    resultado_markdown = run_rag_crew(pergunta)
    resultado_html = Markup(markdown.markdown(resultado_markdown))
    return render_template('result.html', title=f"Resposta para: {pergunta}", pergunta=pergunta, resultado=resultado_html)

@app.route('/generate_procedure', methods=['POST'])
def generate_procedure():
    topic = request.form['topic']
    if not topic:
        return render_template('index.html', error="Por favor, digite um tópico para o procedimento.")
    if not knowledge_base_chunks:
         return render_template('result.html', title="Erro", pergunta=topic, resultado="<p><b>Erro:</b> Nenhum documento PDF encontrado na pasta 'knowledge_base'.</p>")
    resultado_markdown = run_procedure_crew(topic)
    resultado_html = Markup(markdown.markdown(resultado_markdown))
    return render_template('result.html', title=f"Novo Procedimento: {topic}", pergunta=topic, resultado=resultado_html)

if __name__ == '__main__':
    app.run(debug=True, port=5001)