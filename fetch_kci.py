import requests
import xml.etree.ElementTree as ET
import csv
import os


# 설정
BASE_URL = "https://open.kci.go.kr/oai/request"
OUTPUT_CSV = "data/papers.csv"
TARGET_COUNT = 100  # 목표 논문 수


# XML 네임스페이스
NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


# 논문 1편 파싱
def parse_record(record):
    meta = record.find(".//oai:metadata", NS)
    if meta is None:
        return None

    def get_all(tag):
        return [e.text.strip() for e in meta.findall(f".//dc:{tag}", NS) if e.text]

    title = get_all("title")
    abstract = get_all("description")
    creators = get_all("creator")
    subjects = get_all("subject")
    dates = get_all("date")
    languages = get_all("language")

    return {
        "title": " | ".join(title),
        "abstract": " ".join(abstract),
        "authors": " | ".join(creators),
        "keywords": " | ".join(subjects),
        "year": dates[0] if dates else "",
        "language": " | ".join(languages),
    }


# 수집 실행
def collect_papers():
    os.makedirs("data", exist_ok=True)

    papers = []
    params = {
        "verb": "ListRecords",
        "set": "ARTI",
        "metadataPrefix": "oai_dc",
    }

    resumption_token = None

    while len(papers) < TARGET_COUNT:
        if resumption_token:
            params = {
                "verb": "ListRecords",
                "resumptionToken": resumption_token
            }

        print(f"요청 중... 현재 수집: {len(papers)}편")
        r = requests.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()

        root = ET.fromstring(r.text)

        records = root.findall(".//oai:record", NS)
        for record in records:
            paper = parse_record(record)
            if paper:
                papers.append(paper)
                if len(papers) >= TARGET_COUNT:
                    break

        token_elem = root.find(".//oai:resumptionToken", NS)
        if token_elem is None or not token_elem.text:
            break
        resumption_token = token_elem.text

    return papers


# CSV 저장
def save_csv(papers):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["title", "abstract", "authors", "keywords", "year", "language"]
        )
        writer.writeheader()
        writer.writerows(papers)

    print(f"완료: {OUTPUT_CSV} 에 {len(papers)}편 저장됨")


# 실행
if __name__ == "__main__":
    papers = collect_papers()
    save_csv(papers)
