import os
from dotenv import load_dotenv
from anthropic import AnthropicBedrock
from openai import OpenAI
import subprocess
import json

# 환경 변수 로드
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_SESSION_TOKEN = os.getenv('AWS_SESSION_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
BEDROCK_ANTHROPIC_MODEL_NAME = os.getenv('BEDROCK_ANTHROPIC_MODEL_NAME')
OPENAI_MODEL_NAME = "gpt-4.1"


def run_claude(system_text, user_text):
    """
    AWS Bedrock을 통해 Claude 모델을 실행하여 응답을 생성합니다.
    
    Args:
        system_text (str): 시스템 프롬프트 텍스트
        user_text (str): 사용자 입력 텍스트
    
    Returns:
        anthropic.types.Message: Claude 모델의 응답 메시지 객체
    """
    client = AnthropicBedrock(
        aws_access_key=AWS_ACCESS_KEY_ID,
        aws_secret_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        aws_region = 'us-west-2'
        )
    
    message = client.messages.create(
        model = BEDROCK_ANTHROPIC_MODEL_NAME,
        max_tokens=	6000,
        temperature=0,
        system=system_text,
        stop_sequences=['</json>', '</output>'],
        messages=[{
            "role": "user",
            "content": user_text
        }])
    
    return message


def run_openai_chat_completions(system_text, user_text):
    """
    OpenAI의 chat completions API를 사용하여 채팅 응답을 생성합니다.
    
    Args:
        system_text (str): 시스템 프롬프트 텍스트
        user_text (str): 사용자 입력 텍스트
    
    Returns:
        openai.types.chat.ChatCompletion: OpenAI chat completions API의 응답 객체
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    response = client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_text}
                ]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}]
            }
        ],
        response_format={"type": "text"},
        temperature=0.5,
        max_completion_tokens=2048,
        stop=["</json>"],
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return response


def google_websearch(query, pageSize):
    """
    Google Discovery Engine을 사용하여 웹 검색을 수행합니다.
    
    Args:
        query (str): 검색할 쿼리 문자열
        pageSize (int): 반환할 검색 결과의 개수
    
    Returns:
        str: 검색 결과를 포맷팅한 문자열. 각 결과는 제목, 링크, 스니펫을 포함합니다.
             검색 결과가 없거나 오류가 발생한 경우 'No Search Result'를 반환합니다.
    
    Note:
        - Google Cloud Discovery Engine API를 curl 명령어로 호출합니다.
        - gcloud auth print-access-token을 사용하여 인증합니다.
        - 쿼리 확장 및 맞춤법 검사가 자동으로 활성화됩니다.
    """
    curl_command = '''curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
"https://discoveryengine.googleapis.com/v1alpha/projects/908271536747/locations/global/collections/default_collection/engines/miridih-aippt-test_1727068447441/servingConfigs/default_search:search" \
-d "{\\"query\\": \\"%s\\", \\"pageSize\\": %d, \\"queryExpansionSpec\\": {\\"condition\\": \\"AUTO\\"}, \\"spellCorrectionSpec\\": {\\"mode\\": \\"AUTO\\"}, \\"contentSearchSpec\\": {\\"snippetSpec\\": {\\"returnSnippet\\": true}}}"
'''
    try:
        command = curl_command % (query, pageSize)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            result = stdout.decode('utf-8')
            result = json.loads(result)

            search_result = """"""

            if "results" in result.keys() and result["results"]:  # results가 존재하고 비어있지 않은 경우
                for i in range(len(result['results'])):
                    result_title = result['results'][i]['document']['derivedStructData']['title']
                    result_snippet = result['results'][i]['document']['derivedStructData']['snippets'][0]['snippet']
                    result_link = result['results'][i]['document']['derivedStructData']['link']

                    search_result += f"- Title: {result_title}\n- Link: {result_link}\n- Snippet: {result_snippet}\n\n"
                search_result = search_result.rstrip('\n')
                return search_result
            else:
                return 'No Search Result'
        else:
            print(f"Error executing curl command: {stderr.decode('utf-8')}")
            return 'No Search Result'
    except Exception as e:
        print(f"Error in google_websearch: {str(e)}")
        return 'No Search Result' 