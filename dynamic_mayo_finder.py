# dynamic_mayo_finder.py

import requests
from bs4 import BeautifulSoup

CONDITION_DRUG_MAP = {
    "alzheimer": [
        "donepezil-oral-route/description/drg-20067235",
        "memantine-oral-route/description/drg-20035847",
        "rivastigmine-oral-route/description/drg-20066908"
    ],
    "hypertension": [
        "amlodipine-oral-route/description/drg-20061782",
        "lisinopril-oral-route/description/drg-20069129"
    ],
    "diabetes": [
        "metformin-oral-route/description/drg-20067074",
        "insulin-injection-route/description/drg-20073846"
    ]
}

def scrape_mayo_drug(drug_slug):
    base_url = "https://www.mayoclinic.org/drugs-supplements/"
    url = f"{base_url}{drug_slug}"

    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.find("h1").text.strip()

    paragraphs = soup.find_all("p")
    content = "\n".join([p.text.strip() for p in paragraphs if p.text.strip()])

    return {
        "url": url,
        "title": title,
        "content": content[:1500]
    }

def get_drugs_for_condition(condition):
    condition = condition.lower()
    drug_slugs = CONDITION_DRUG_MAP.get(condition, [])
    results = []

    for slug in drug_slugs:
        scraped = scrape_mayo_drug(slug)
        results.append(scraped)

    return results
