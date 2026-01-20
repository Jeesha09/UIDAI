import streamlit as st
import google.generativeai as genai
from pandasai import Agent
from pandasai.llm import LLM
from pandasai.core.prompts.base import BasePrompt
from dashboard_context import get_page_context

HARDCODED_API_KEY = "AIzaSyAItQIgASLO5fdRGitBvd2PEUuRMHcgOn0"

class GeminiLLM(LLM):
    """Custom LLM wrapper for Google Gemini to work with PandasAI 3.0.0"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key=api_key)
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel('gemini-2.5-flash')
    
    def call(self, instruction: BasePrompt, context=None) -> str:
        """Execute the LLM with given prompt"""
        prompt_text = str(instruction)
        response = self._model.generate_content(prompt_text)
        return response.text
    
    @property
    def type(self) -> str:
        return "google-gemini"

class AadhaarChatbot:
    def __init__(self, df_enrol, df_bio, df_demo):
        self.api_key = HARDCODED_API_KEY
        self.dfs = [df_enrol, df_bio, df_demo]
        self._model = None
        self._llm_pandas = None

    def _get_gemini_model(self):
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel("gemini-2.5-flash")
        return self._model

    def _get_pandas_llm(self):
        if self._llm_pandas is None:
            self._llm_pandas = GeminiLLM(api_key=self.api_key)
        return self._llm_pandas

    def _decide_intent(self, user_query):
        """
        Decides if the user wants a CALCULATION (Data) or EXPLANATION (Context).
        """
        system_prompt = """
        You are a classifier. Reply ONLY with 'DATA' or 'CONTEXT'.
        1. DATA: calculating numbers, summing, finding max/min, querying CSVs.
        2. CONTEXT: explaining charts, definitions, colors, or methodology.
        User Query:
        """
        
        try:
            model = self._get_gemini_model()
            response = model.generate_content(system_prompt + user_query)
            return response.text.strip().upper()
        except:
            return "DATA"

    def _answer_with_context(self, user_query, current_page):
        page_info = get_page_context(current_page)
        
        prompt = f"""
        You are a dashboard assistant. The user is on the '{current_page}' page.
        
        Here is the context for the charts on this page:
        {page_info}
        
        User Question: {user_query}
        
        Answer strictly based on the context provided above.
        """
        
        try:
            model = self._get_gemini_model()
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I couldn't generate an explanation. Error: {str(e)}"

    def _answer_with_data(self, user_query):
        try:
            # Step 1: Get the RAW Answer from PandasAI
            agent = Agent(self.dfs, config={"llm": self._get_pandas_llm()})
            
            query = f"""
            I have 3 datasets:
            1. Enrollments (cols: age_0_5, age_18_greater, district, state)
            2. Biometrics (cols: bio_age_5_17, district, state)
            3. Demographics (cols: demo_age_5_17, district, state)
            
            Question: {user_query}
            """
            
            raw_response = agent.chat(query)
            
            # Step 2: The "Humanizer" Step
            # We take that messy table and ask Gemini to write a sentence.
            humanizer_prompt = f"""
            The user asked: "{user_query}"
            
            The data analysis tool returned this raw result:
            "{raw_response}"
            
            Please rewrite this result as a clear, natural English sentence or a bulleted list. 
            Do not just paste the table. Explain what the numbers mean.
            """
            
            model = self._get_gemini_model()
            final_response = model.generate_content(humanizer_prompt)
            return final_response.text
            
        except Exception as e:
            return f"I tried to calculate that, but ran into an error: {str(e)}"

    def ask(self, user_query, current_page):
        intent = self._decide_intent(user_query)
        if "CONTEXT" in intent:
            return self._answer_with_context(user_query, current_page)
        else:
            return self._answer_with_data(user_query)