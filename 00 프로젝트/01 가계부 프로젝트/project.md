# 가계부 프로젝트

## 개요
- 데이터분석 라이브러리 `numpy`, `pandas` 그리고 pandas의 `Series` 와 `DataFrame` 객체, 그리고 `matplotlib` 시각화 모듈에 대해 공부하며 가계부 프로젝트에 대한 아이디어가 떠올랐다.
- 일반 은행과 카드 결제사 앱, 그리고 Toss나 Bank Salad를 비롯한 금융 관리 통합 앱을 통해 자동으로 기록되는 결제 내역(소비/지출, 수입)을 바탕으로 가계부를 확인하며 관리하고 있다. 그럼에도, 과소비를 방지하고 소비와 지출에 대한 내역을 매일 재검토하고자 하는 의도에서 직접 결제한 내역과 수입 내역을 Notion에 표 데이터베이스를 만들어 작성하고 있다.

<br>
<div align="center">
    <img src="https://user-images.githubusercontent.com/71310074/196048716-c56b8a54-22d5-446d-b360-4a2a094857f6.png" width="200"> <img src="https://user-images.githubusercontent.com/71310074/196048856-84d91a44-0db6-4a87-950a-f09758ee5a1f.png" width="348">

</div>

<br>

## 서비스 분석

### 뱅크 샐러드

<div align="center"><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2FkMCBC%2Fbtq6udfBN3T%2FIiemhzYHmNe6I0Mqast2p1%2Fimg.png" width="150">  <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2FQivFa%2Fbtq6vhIsCgu%2Fx7uZeb0RQXIBO8Mkd2mmK0%2Fimg.png" width="150"> 
</div>


### 이번달 수입/지출 주 단위 요약, 카테고리별 비율 시각화, 일별 추이 그래프 시각화

<div align="center">
<img src="https://user-images.githubusercontent.com/71310074/196050115-77b939f8-b63a-4122-96eb-e77ac765ac3b.png" width="153"> <img src="https://user-images.githubusercontent.com/71310074/196050146-88a52a3b-d411-4ef8-b36a-6aae30eb92dc.png" width="160"> <img src="https://user-images.githubusercontent.com/71310074/196050167-3506b18e-6e19-448f-be4a-a4b9aba3f11c.png" width="160"> <img src="https://user-images.githubusercontent.com/71310074/196050045-4822b204-b844-4666-a9dd-8126ee6e7a2b.png" width="400">
</div>

### 예산 계획하기 기능

<div align="center">
<img src="https://user-images.githubusercontent.com/71310074/196050704-8172e414-239b-4682-b79a-005bba953e87.png" width="300"> <img src="https://user-images.githubusercontent.com/71310074/196050740-f9472b6f-6629-4194-b189-bba7b2222f4b.png" width="200"> <img src="https://user-images.githubusercontent.com/71310074/196050842-ad21bafa-0e08-44ae-94c2-0fbc80c0a36f.png" width="200"> <img src="https://user-images.githubusercontent.com/71310074/196050861-76799165-062e-403c-a81a-54d4ecbc4cb2.png" width="200"> <img src="https://user-images.githubusercontent.com/71310074/196050891-e27ef090-37f5-4808-9197-b3e17107b668.png" width="200"> <img src="" width="200">
</div>

<br>

- 우선 가계부 조회 필터링은 다음과 같이 5개가 존재한다.
    1. 기간 ( 전체 / 월간 / 주간 / 연간 / 최근3개월 / 최근6개월 / 직접선택(시작일,마지막일) )
    2. 결제수단 (모든 결제 수단 / 현금 / ...각종 연결되어 있는 계좌, 페이머니 등 )
    3. 카테고리 ( 지출 / 수입 / 이체 )
- 이중에서도 가장 중요한 지출 카테고리의 세부 카테고리를 살펴보자.
    1. 식비 -> 한식/중식/일식/양식/아시아음식/뷔페/고기/치킨/피자/패스트푸드/배달/식재료
    2. 카페/간식 -> 커피,음료/베이커리/디저트,떡/도넛,핫도그/아이스크림,빙수/기타간식
    3. 술/유흥 -> 맥주,호프/이자카야/와인/바/요리주점/민속주점/유흥시설
    4. 생활 -> 생필품/편의점/마트/생활서비스/세탁/목욕/가구,가전
    5. 온라인쇼핑 -> 인터넷쇼핑/홈소핑/결제,충전/앱스토어/서비스구독
    6. 패션/쇼핑 -> 패션/신발/아울렛,몰/스포츠의류/백화점
    7. 뷰티/미용 -> 화장품/헤어샵/미용관리/미용용품/네일성형외과/피부과
    8. 교통 -> 택시/대중교통/철도/시외버스
    9. 자동차 -> 주유/주차/세차/통행료/할부,리스/장비,수리/차량보험/대리운전
    10. 


- 가계부의 카테고리는 크게 지출/수입/이체 로 나뉘어져 있다.
- 


<br>

---

# 가계부 분석 및 통계 시각화 프로그램 

## 필요 데이터

1. 유저 메타 데이터
- 성별
- 나이대
- 거주지역
- 신분 (학생/직장인)
- 자산, 신용 등급, 연결된 계좌 등 금융/계좌 관련 내역

2. 데이터 항목
- 상품이름 
- 지출or소비
- 액수
- 카테고리 (식비, 의류/생활, 교통, 문화/여가, 운동, 오락, 카페/간식)
- 날짜
- 메모
- 상품 구매 지역

<br>

## 기능 사항

### 서비스 기능 사항
- 주별 지출 요약과 합계를 나타낸다.
- matplotlib의 파이차트를 이용해 카테고리별 지출 현황을 시각화하여 표현한다.
- 꺾은선 그래프를 이용해, 일별 지출과 소비 액수의 추이를 시각화한다.
- 꺾은선 그래프(색이 다른 두 선 표현) 2개월 간의 소비지출 추이 대비를 시각화한다.
- 새로운 소비/지출 데이터를 추가한다.
    - 새로둔 데이터 추가 시 연결되어 있는 지출 통계에도 변경 내용을 반영한다. (예- 카테고리별 지출 비율이 달라지며 그래프도 함께 변화)