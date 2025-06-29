from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import datetime
import json
import re
import os

def wait_for_element(driver, selector, timeout=5):  # 셀레니움에서 특정 요소가 로드될 때까지 대기
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
        return True
    except:
        return False
        
def load_card_page(driver, card_id: int) -> bool:  # 카드 상세 페이지 로드 및 정상 여부 반환
    url = f"https://www.card-gorilla.com/card/detail/{card_id}"

    driver.get(url)
    if not wait_for_element(driver, ".tit .card"):
        print(f"[!] 페이지 로딩 실패 (ID: {card_id})")
        return False
    return True
   
def expand_dl_sections(driver):  # 카드 혜택/유의사항 섹션 모두 펼치기
    script = """
    const dls = document.querySelectorAll("div.lst.bene_area > dl");
    let count = 0;
    dls.forEach(dl => {
        if (!dl.classList.contains("on")) {
            dl.classList.add("on");

            // dt 클릭 이벤트 강제로 발생시켜 렌더링 유도
            const dt = dl.querySelector("dt");
            if (dt) {
                dt.dispatchEvent(new MouseEvent('click', { bubbles: true }));
            }

            count += 1;
        }
    });
    return count;
    """
    try:
        count = driver.execute_script(script)
        print(f"[+] 혜택 영역 {count}개 확장 완료")
    except Exception as e:
        print("[!] 혜택 영역 확장 실패: ", e)
    
def extract_card_details(driver):  # 카드 혜택/유의사항 정보 추출
    benefits, cautions = [], []
    try:
        # 혜택/유의사항이 확장된 dl 요소만 추출
        dl_elements = driver.find_elements(By.CSS_SELECTOR, 'div.lst.bene_area > dl.on')
        for dl in dl_elements:
            try:
                dt = dl.find_element(By.TAG_NAME, "dt")
                benefit_type = dt.find_element(By.CSS_SELECTOR, "p.txt1").text.strip()
                summary = dt.find_element(By.TAG_NAME, "i").get_attribute("innerText").strip().replace('\n', ' ')
                
                p_tags = dl.find_elements(By.CSS_SELECTOR, "dd > div.in_box > p")
                details, is_caution = [], "유의사항" in benefit_type
                
                for p in p_tags:
                    text = p.text.strip()
                    if not text or text == "<br>":
                        continue
                    clean  = text.lstrip("-").strip()
                    (cautions if is_caution or "유의사항" in text else details).append(clean)
                    
                if not is_caution:
                    benefits.append({"type": benefit_type, "summary":summary, "details":details})
            except Exception as e:
                print("[!] 혜택 파싱 실패: ", e)
    except Exception as e:
        print("[!] 혜택 여역 추출 실패: ", e)
    return benefits, cautions
        
def extract_card_info(driver, card_id: int) -> dict:  # 카드 상세 정보(혜택, 연회비 등) 추출
    data = {"card_id": card_id, "card_url": f"https://www.card-gorilla.com/card/detail/{card_id}"}
    
    # 카드명, 카드사, 발급여부, 브랜드
    selectors = {
        "card_name": (By.CSS_SELECTOR, ".tit .card"),
        "issuer": (By.CLASS_NAME, "brand"),
        "inactive": (By.CLASS_NAME, "inactive"),
        "brands": (By.CSS_SELECTOR, ".c_brand span")
    }
    
    for key, (by, sel) in selectors.items():
        try:
            elems = driver.find_elements(by, sel)
            if key == "inactive":
                data[key] = elems[0].text.strip() if elems else False
            elif key == "brands":
                data[key] = [e.text.strip() for e in elems if e.text.strip()]
            else:
                data[key] = elems[0].text.strip() if elems else None
        except:
            data[key] = [] if key == "brands" else None
        
        # 연회비
    try:
        fees = driver.find_elements(By.CSS_SELECTOR, ".in_out span")
        data["annual_fee"] = {
            "domestic": fees[0].text.strip() if len(fees) > 0 else None,
            "international": fees[1].text.strip() if len(fees) > 1 else None
        }
    except:
        data["annual_fee"] = {"domestic": None, "international": None}
        
    try:
        section = driver.find_element(By.CLASS_NAME, "bnf2")
        for dl in section.find_elements(By.TAG_NAME, "dl"):
            dt = dl.findelement(By.TAG_NAME, "dt").text.strip()
            if "전월실적" in dt:
                dd = dl.find_element(By.TAG_NAME, "dd")
                raw = dd.text.strip().replace("\n", "")
                data["required_spending"] = raw
                match = re.search(r"(\d+)", raw)
                data["required_spending_amount"] = int(match.group(1)) if match else None
                break
        else:
            data["required_spending"] = data["required_spending_amount"] = None
    except:
        data["required_spending"] = data["required_spending_amount"] = None
            
    data["benefits"], data["caution"] = extract_card_details(driver)
    return data

def save_card_json(data: dict, output_root="./crawling/beomseok/cards"):  # 카드 정보를 JSON 파일로 저장
    issuer = data.get("issuer", "unknown").strip().replace(" ", "_")
    
    # 디렉토리 경로: cards/현대카드/
    output_dir = os.path.join(output_root, issuer)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{issuer}_{data['card_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"저장 완료: {path}")
    
def get_card_info(card_id: int, save=True) -> dict:  # 카드 상세 정보 크롤링 및 저장
    options = Options()
    for arg in ["--headless", "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"]:
        options.add_argument(arg)
    driver = webdriver.Chrome(options=options)
    
    try:
        if not load_card_page(driver, card_id):
            return {}
        
        expand_dl_sections(driver)
        data = extract_card_info(driver, card_id)
        
        if save:
            save_card_json(data)

        return data
    finally:
        driver.quit()

def save_all_cards(start_id=1, end_id=2857, log_path="./card_crawl_log.txt"):  # 전체 카드 ID 범위에 대해 일괄 크롤링 및 저장
    with open(log_path, "w", encoding="utf-8") as log:
        for card_id in range(start_id, end_id + 1):
            print(f"\n[+] 카드 ID {card_id} 처리 중...")
            try:
                data = get_card_info(card_id, save=True)
                log.write(f"{card_id} {'CLEAR' if data.get('card_name') else 'FAIL'}\n")
            except Exception as e:
                print(f"[!] 처리 실패 (ID: {card_id}: {e})")
                log.write(f"{card_id} FAIL\n")
            time.sleep(0.5)


if __name__ == "__main__":
    start = time.time()
    save_all_cards(1, 2857)
    end = time.time()
    elapsed = str(datetime.timedelta(seconds=end - start)).split(".")[0]
    print(f"\n전체 소요 시간: {elapsed}")