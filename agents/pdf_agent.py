import os
import json
import re
from io import BytesIO
from typing import Any, Dict, List
import logging
from datetime import datetime

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

load_dotenv()

class PDFAgent:
    """
    Enhanced PDF Agent that:
    - Processes PDF files from raw bytes
    - Extracts text and structured data based on business intent
    - Makes intelligent decisions based on extracted content
    - Integrates with shared memory and action routing
    """

    def __init__(self, temperature: float = 0.0, shared_memory=None):
        self.shared_memory = shared_memory
        self.logger = logging.getLogger(__name__)
        
        try:
            self.llm = ChatGroq(
                model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
                temperature=temperature,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Groq LLM: {e}")
            raise

        self.prompts = {
            "invoice": PromptTemplate(
                input_variables=["text"],
                template="""
                Analyze this invoice text and extract structured data in JSON format:
                
                Text: {text}
                
                Extract and return ONLY a valid JSON object with these fields:
                {{
                    "invoice_number": "string",
                    "date": "string",
                    "vendor": "string",
                    "line_items": [
                        {{"description": "string", "quantity": number, "unit_price": number, "total": number}}
                    ],
                    "subtotal": number,
                    "tax": number,
                    "invoice_total": number,
                    "payment_terms": "string"
                }}
                
                If information is not found, use null for strings and 0 for numbers.
                """
            ),
            
            "regulation": PromptTemplate(
                input_variables=["text"],
                template="""
                Analyze this regulatory document and extract policy-related information in JSON format:
                
                Text: {text}
                
                Extract and return ONLY a valid JSON object with these fields:
                {{
                    "document_type": "string",
                    "policy_mentions": ["list of policy keywords found like GDPR, FDA, HIPAA, etc."],
                    "compliance_requirements": ["list of compliance requirements mentioned"],
                    "effective_date": "string",
                    "regulatory_body": "string",
                    "risk_level": "low|medium|high"
                }}
                
                Look specifically for keywords: GDPR, FDA, HIPAA, SOX, PCI-DSS, ISO, OSHA, EPA
                """
            ),
            
            "general": PromptTemplate(
                input_variables=["text"],
                template="""
                Analyze this document and provide a summary in JSON format:
                
                Text: {text}
                
                Extract and return ONLY a valid JSON object with these fields:
                {{
                    "document_type": "string",
                    "key_topics": ["list of main topics"],
                    "summary": "brief summary in 2-3 sentences",
                    "entities": ["list of organizations, people, dates mentioned"],
                    "action_items": ["list of any action items or next steps mentioned"]
                }}
                """
            )
        }

    def process(self, raw_bytes: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method that:
        1. Extracts text from PDF
        2. Processes based on intent
        3. Makes action decisions
        4. Logs to shared memory
        """
        try:
            # Extract metadata
            intent = metadata.get("intent", "general").lower()
            source_id = metadata.get("source_id", f"pdf_{datetime.now().timestamp()}")
            
            # Extract text from PDF
            full_text = self._extract_text_from_bytes(raw_bytes)
            
            if not full_text.strip():
                return self._create_error_response("No text could be extracted from PDF", source_id)
            
            # Truncate text to avoid token limits (keep first 3000 chars for better context)
            processed_text = full_text[:3000]
            
            # Process based on intent
            extracted_data = self._process_by_intent(processed_text, intent)
            
            # Make action decision
            action_suggestion = self._determine_action(extracted_data, intent)
            
            # Prepare response
            response = {
                "source": "pdf_agent",
                "source_id": source_id,
                "timestamp": datetime.now().isoformat(),
                "intent": intent,
                "text_length": len(full_text),
                "data": extracted_data,
                "action_suggestion": action_suggestion,
                "status": "success"
            }
            
            # Log to shared memory
            self._log_to_memory(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            return self._create_error_response(str(e), metadata.get("source_id", "unknown"))

    def _extract_text_from_bytes(self, raw_bytes: bytes) -> str:
        try:
            pdf_stream = BytesIO(raw_bytes)
            reader = PdfReader(pdf_stream)
            
            text_parts = []
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                except Exception as e:
                    self.logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                    continue
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            self.logger.error(f"Error reading PDF: {e}")
            raise Exception(f"Failed to extract text from PDF: {e}")

    def _process_by_intent(self, text: str, intent: str) -> Dict[str, Any]:
        try:
            # Select appropriate prompt
            prompt_template = self.prompts.get(intent, self.prompts["general"])
            
            # Create chain and run
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            llm_response = chain.run(text=text).strip()
            
            # Clean and parse JSON response
            json_response = self._extract_json_from_response(llm_response)
            
            if json_response:
                return json_response
            else:
                # Fallback if JSON parsing fails
                return {
                    "raw_response": llm_response,
                    "extraction_method": "fallback",
                    "text_snippet": text[:500] + "..." if len(text) > 500 else text
                }
                
        except Exception as e:
            self.logger.error(f"Error in LLM processing: {e}")
            return {
                "error": str(e),
                "extraction_method": "failed",
                "text_snippet": text[:500] + "..." if len(text) > 500 else text
            }

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        try:
            # Try direct JSON parsing first
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON within the response
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            
            return None

    def _determine_action(self, data: Dict[str, Any], intent: str) -> Dict[str, str]:
        default_action = {"action": "archive", "target": "database"}
        
        try:
            if intent == "invoice":
                invoice_total = float(data.get("invoice_total", 0))
                if invoice_total > 10000:
                    return {
                        "action": "flag_compliance", 
                        "target": "risk_alert",
                        "reason": f"High value invoice: ${invoice_total:,.2f}"
                    }
                return {
                    "action": "process_payment", 
                    "target": "accounting_system",
                    "reason": f"Standard invoice: ${invoice_total:,.2f}"
                }
                
            elif intent == "regulation":
                policy_mentions = data.get("policy_mentions", [])
                risk_level = data.get("risk_level", "low")
                
                if policy_mentions or risk_level in ["medium", "high"]:
                    return {
                        "action": "flag_compliance", 
                        "target": "risk_alert",
                        "reason": f"Regulatory compliance required: {', '.join(policy_mentions)}"
                    }
                return {
                    "action": "file_regulatory", 
                    "target": "compliance_database",
                    "reason": "Standard regulatory document"
                }
                
            else:  # general
                action_items = data.get("action_items", [])
                if action_items:
                    return {
                        "action": "create_tasks", 
                        "target": "task_management",
                        "reason": f"Document contains {len(action_items)} action items"
                    }
                    
        except Exception as e:
            self.logger.error(f"Error determining action: {e}")
        
        return default_action

    def _log_to_memory(self, response: Dict[str, Any]):
        """Log processing results to shared memory"""
        if self.shared_memory:
            try:
                memory_entry = {
                    "agent": "pdf_agent",
                    "timestamp": response["timestamp"],
                    "source_id": response["source_id"],
                    "intent": response["intent"],
                    "action_suggested": response["action_suggestion"],
                    "data_summary": {
                        "text_length": response["text_length"],
                        "extraction_success": response["status"] == "success"
                    }
                }
                self.shared_memory.store("pdf_processing", memory_entry)
            except Exception as e:
                self.logger.error(f"Error logging to shared memory: {e}")

    def _create_error_response(self, error_msg: str, source_id: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "source": "pdf_agent",
            "source_id": source_id,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": error_msg,
            "data": {},
            "action_suggestion": {"action": "manual_review", "target": "error_queue"}
        }

    def validate_pdf(self, raw_bytes: bytes) -> bool:
        """Validate that the bytes represent a valid PDF"""
        try:
            pdf_stream = BytesIO(raw_bytes)
            PdfReader(pdf_stream)
            return True
        except Exception:
            return False