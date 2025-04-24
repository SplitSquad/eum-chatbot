import requests
import fitz
import os
import pdfplumber
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
from dotenv import load_dotenv
load_dotenv()  # .env 파일 자동 로딩
import easyocr
import glob  # 이미지 파일 목록 가져오기용

############################################# pdf 텍스트 추출
def text_extraction(PDF_FILE_PATH):
    
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

    ###### first_ocr
    image_files = sorted(glob.glob('./data/*.png')) # ✅ 모든 PNG 파일 경로 가져오기 (정렬 포함)
    first_ocr_text_list = []         # ✅ 결과 저장용 리스트
    # ✅ 사용자 설정
    api_key = 'K85417667888957'  # 발급받은 API 키
    language = 'eng'              # 'kor'도 가능
    overlay = False               # 단어 위치 좌표 필요 여부
    # ✅ payload 구성
    payload = {
        'isOverlayRequired': overlay,
        'apikey': api_key,
        'language': language,
    }
    # ✅ OCR API 반복 호출
    for filename in image_files:
        with open(filename, 'rb') as f:
            response = requests.post(
                'https://api.ocr.space/parse/image',
                files={'filename': f},
                data=payload,
            )
        result = response.json()
        parsed_text = result['ParsedResults'][0]['ParsedText']
        first_ocr_text_list.append(parsed_text.strip())
    # ✅ 결과 출력
    for i, text in enumerate(first_ocr_text_list):  
        print(f"\n📸 first OCR 결과 Page {i+1}:\n{text}")
    first_ocr_text = "------".join(first_ocr_text_list) 

    ###### second_ocr
    reader = easyocr.Reader(['ko'])  # 한국어 + 영어, CPU 모드
    second_ocr_lines = []
    for filename in image_files:
        result = reader.readtext(filename)
        for bbox, text, confidence in result:
            second_ocr_lines.append(f"[{confidence:.2f}] {text}")
    # 줄바꿈 포함하여 전체 텍스트 결합
    second_ocr_text = "------".join(second_ocr_lines)

    ###### third_ocr


    # ✅ PDF에서 추출된 text_list와 결합
    parsed_text = (
        "📄Text extracted from pdf \n"
        + "\n---\n".join(text_list)
        + "\n\n\n\n 📸Text from image extracted from pdf \n"
        + " first OCR MODEL \n"
        + first_ocr_text
        + " second OCR MODEL \n"
        + second_ocr_text
    )

    print("parsed_text",parsed_text)

    # ✅ OCR 이후, data 폴더 내 PNG 이미지 삭제
    for file in glob.glob('./data/*.png'):
        os.remove(file)

    return parsed_text
############################################# pdf 텍스트 추출


############################################# pdf 카테고리 분류
def classification(parsed_text) :
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
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
    1. Your role is to categorize the PDFs sent by users.
    2. Return only a JSON object with the following format:
    "input": "...",
    "output": "..."  // One of: agentic-calendar, agentic-resume, agentic-create, agentic-visa, general

    
    3. Do not add any explanations. Do not return markdown. Do not add commentary.

    input : "2025년 3월 4일 ~ 10일: 제1학기 수강신청 변경 기간"
    output : "agentic-calendar"

    input : "2025년 6월 30일: 제1학기 성적제출 마감일"
    output : "agentic-calendar"
    
    input : "2026년 1월 6일 ~ 10일: 2026학년도 전과 신청"
    output : "agentic-calendar"

    input : "제2학기 개강: 2025년 9월 1일"
    output : "agentic-calendar"

    input : "제2학기 재학생 등록: 8월 25일 ~ 29일"
    output : "agentic-calendar"

    input : "2026년 2월 28일: 동계휴가 종료"
    output : "agentic-calendar"

    input : "제1학기 보강 기간: 6월 17일 ~ 23일"
    output : "agentic-calendar"

    


    input : "이력서"
    output : "agentic-resume"

    input : "성명: 홍길동 / 생년월일: 1990.01.01"
    output : "agentic-resume"

    input : "학력: 2010~2014 서울대학교 컴퓨터공학과 졸업"
    output : "agentic-resume"

    input : "경력: 2015~2020 Naver Corp. 백엔드 개발자"
    output : "agentic-resume"

    input : "자격증: 정보처리기사"
    output : "agentic-resume"

    input : "자기소개서"
    output : "agentic-resume"

    input : "학위: 석사 / 전공: 인공지능"
    output : "agentic-resume"

    input : "경력사항 요약"
    output : "agentic-resume"


    

    input : "2024 대전 이시 축제 개최 안내"
    output : "agentic-create"

    input : "행사일시: 2024년 8월 9일 ~ 17일"
    output : "agentic-create"

    input : "K-pop 콘서트, 과학 전시 체험, 퍼레이드"
    output : "agentic-create"

    input : "바로가기 / 대전시 공식 행사"
    output : "agentic-create"

    input : "행사 포스터 / 장소: 옛 충남도청"
    output : "agentic-create"

    input : "출연진: 화사, 제시, 다비치 등"
    output : "agentic-create"

    input : "과학기술 테마파크 / 체험 부스 운영"
    output : "agentic-create"

    input : "이무진, 박지현, 장민호 등 출연"
    output : "agentic-create"

    

    input : "IMMIGRANT VISA / IV Case Number: 00200416000201"
    output : "agentic-visa"

    input : "Given name: HAPPY / Nationality: GBR"
    output : "agentic-visa"

    input : "Issue Date: 23DEC2004 / Passport Number: 555123ABC"
    output : "agentic-visa"

    input : "US CONSULATE GENERAL - LONDON"
    output : "agentic-visa"

    input : "Visa Type: TRAVELER"
    output : "agentic-visa"

    input : "Date of Birth: 05FEB1965 / Status: IV On"
    output : "agentic-visa"

    input : "SERVES I-551 / Immigrant Visa Status"
    output : "agentic-visa"

    input : "Case No: 00000473 / Category: F11"
    output : "agentic-visa"

    input : "Immigration Barcode: 555123ABC6GBR6502056F04122361FLNDOOAMS803085"
    output : "agentic-visa"

    input : "Residence: United Kingdom / Entry Code: IV"
    output : "agentic-visa"

    

    intput : All answers other than the above examples
    output : general


    
    Only return the JSON object like above.
    """
    
    ),
        ("user", "{input}")
    ])

    chain = prompt | llm | parser

    def parse_product(description: str) -> dict:
            result = chain.invoke({"input": description})
            print(json.dumps(result, indent=2, ensure_ascii=False))

            return json.dumps(result, indent=2, ensure_ascii=False)

    description = parsed_text
    print("description\n",description)
    response = parse_product(description)
    response = json.loads(response)
    print("response\n",response)
    
    return response

############################################# pdf 카테고리 분류


############################################# pdf 요약 
def pdf_general(parsed_text) :

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
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
        print(json.dumps(result, indent=2, ensure_ascii=False))

        return json.dumps(result, indent=2, ensure_ascii=False)

    description = parsed_text
    print("description\n",description)
    response = parse_product(description)
    print("response\n",response)

    return response
############################################# pdf 요약

############################################# 메인
# ✅ 예제 PDF 파일 경로
PDF_FILE_PATH = "./학사일정.pdf"
parsed_text = text_extraction(PDF_FILE_PATH) # pdf 추출
Category = classification(parsed_text) #
Category = Category["output"]

if Category == "general" :
    print("general")
    general=pdf_general(parsed_text)
    print("general\n",general)
elif Category == "agentic-calendar" :
    print("agentic-calendar로 분류됨")
elif Category == "agentic-create" :
    print("agentic-create로 분류됨")
elif Category == "agentic-visa" :
    print("agentic-visa로 분류됨")
elif Category == "agentic-resume" :
    print("agentic-resume로 분류됨")
else :
    print("잘못된 카테고리 입니다.")

print("Category\n",Category)