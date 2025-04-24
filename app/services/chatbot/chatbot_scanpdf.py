import requests
import fitz
import os
import pdfplumber
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
from dotenv import load_dotenv
load_dotenv(".env")  # 프로젝트 루트의 .env 파일 로드

# ✅ 예제 PDF 파일 경로
PDF_FILE_PATH = "./학사일정.pdf"

# ✅ 리스트에 저장
text_list = []
with pdfplumber.open(PDF_FILE_PATH) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            text_list.append(text.strip())

# ✅ 출력 예시
for i, page_text in enumerate(text_list):
    print(f"\n📄 Page {i+1} 텍스트:\n{page_text}")

# ✅ 저장할 폴더 생성
os.makedirs('./data', exist_ok=True)

# ✅ PDF → PNG 변환
doc = fitz.open(PDF_FILE_PATH)
for i, page in enumerate(doc):
    img = page.get_pixmap()
    output_path = f"./data/{i}.png"
    img.save(output_path)
    # print(f"📸 저장 완료: {output_path}")

# ✅ 사용자 설정
api_key = 'K85417667888957'  # 발급받은 API 키
filename = './data/0.png'     # OCR할 이미지 경로
language = 'eng'              # 'kor'도 가능
overlay = False               # 단어 위치 좌표 필요 여부

# ✅ payload 구성
payload = {
    'isOverlayRequired': overlay,
    'apikey': api_key,
    'language': language,
}

# ✅ OCR API 요청
with open(filename, 'rb') as f:
    response = requests.post(
        'https://api.ocr.space/parse/image',
        files={'filename': f},
        data=payload,
    )

# ✅ 결과 출력
result = response.json()
# print("result\n",result)

parsed_text = result['ParsedResults'][0]['ParsedText']
print("\n\n 📸 이미지에서 추출된 텍스트 \n ", parsed_text)

parsed_text = "📄Text extracted from pdf \n" + "\n---\n".join(text_list) + "\n\n\n\n 📸Text from image extracted from pdf \n" + parsed_text

############################################# ai응답
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.environ.get("GROQ_API_KEY")
    )

parser = JsonOutputParser(pydantic_object={
    "type": "object",
    "properties": {
        "input": {"type": "string"},
        "output": {"type": "string"},
    }
})
prompt = ChatPromptTemplate.from_messages([
("system", """
1. You are the part that organizes the text.
2. I will give you the text extracted from the pdf.
3. Please organize the text so that it is easy for users to understand.

Return only this JSON:
 
{{ 
"input": "Original text",
"output": "organized text"
}}
"""
 
 ),
    ("user", "{input}")
])

chain = prompt | llm | parser

def parse_product(description: str) -> dict:
        result = chain.invoke({"input": description})
        print(json.dumps(result, indent=2))

        return json.dumps(result, indent=2)

description = parsed_text
print("description\n",description)
response = parse_product(description)
print("response\n",response)
############################################# ai응답