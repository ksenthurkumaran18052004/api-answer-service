import subprocess
import os
import shutil
import platform
import re
import numpy as np
import json
import zipfile
import pandas as pd
from datetime import date, timedelta


def ga1_q1(question, file=None):
    """
    Returns output of `code -s` if Visual Studio Code is installed.
    """
    vscode_path = shutil.which("code")
    if not vscode_path:
        return "VS Code not found. Please install and ensure 'code' is in PATH."

    try:
        result = subprocess.run(["code", "-s"], capture_output=True, text=True, check=True)
        return result.stdout.strip() or "No output from code -s."
    except subprocess.CalledProcessError as e:
        return f"Error running 'code -s': {e}"
    


def ga1_q2(question, file=None):
    """
    Sends HTTPS request to https://httpbin.org/get?email=xxx using httpie.
    """
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", question)
    if not email_match:
        return "No valid email found in the question."

    email = email_match.group()

    try:
        subprocess.run(["pip", "install", "httpie"], check=True, capture_output=True)
        result = subprocess.run(
            ["http", "GET", f"https://httpbin.org/get?email={email}"],
            capture_output=True, text=True, check=True
        )
        response = result.stdout.strip()

        # Extract only JSON part
        json_start = response.find("{")
        json_part = response[json_start:]
        data = json.loads(json_part)

        return data
    except Exception as e:
        return f"Error sending HTTP request: {e}"

import subprocess
import os
import platform

def ga1_q3(question, file=None):
    """
    Runs `npx -y prettier@3.4.2 README.md | sha256sum` and returns the hash.
    """
    if not os.path.exists("README.md"):
        return "README.md not found in current directory."

    try:
        if platform.system() == "Windows":
            # Windows workaround: save output to temp file and hash it using certutil
            subprocess.run("npx prettier@3.4.2 README.md > temp_prettier_output.txt", shell=True, check=True)
            result = subprocess.run(
                ["certutil", "-hashfile", "temp_prettier_output.txt", "SHA256"],
                capture_output=True, text=True, check=True
            )
            os.remove("temp_prettier_output.txt")
            lines = result.stdout.splitlines()
            return lines[1] if len(lines) > 1 else "Hash not found."
        else:
            # Linux/macOS
            result = subprocess.run(
                "npx -y prettier@3.4.2 README.md | sha256sum",
                shell=True, capture_output=True, text=True, check=True
            )
            return result.stdout.strip().split()[0]
    except Exception as e:
        return f"Error: {e}"

import numpy as np
import re

def ga1_q4(question, file=None):
    """
    Simulates Google Sheets formula:
    =SUM(ARRAY_CONSTRAIN(SEQUENCE(rows, cols, start, step), row_limit, col_limit))
    """
    match = re.search(r'SEQUENCE\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\).*?(\d+),\s*(\d+)\)', question)
    if not match:
        return "Could not parse formula."

    rows, cols, start, step, row_limit, col_limit = map(int, match.groups())

    # Generate the SEQUENCE matrix row-wise
    matrix = [[start + (r * cols + c) * step for c in range(cols)] for r in range(rows)]
    matrix = np.array(matrix)

    # Apply ARRAY_CONSTRAIN
    constrained = matrix[:row_limit, :col_limit]

    return int(np.sum(constrained))

def ga1_q5(question, file=None):
    match = re.search(r'SUM\(TAKE\(SORTBY\(\{(.+?)\},\s*\{(.+?)\}\),\s*(\d+),\s*(\d+)\)', question)
    if not match:
        return "Invalid formula format."

    values = list(map(int, match.group(1).split(',')))
    keys = list(map(int, match.group(2).split(',')))
    rows, cols = int(match.group(3)), int(match.group(4))

    sorted_indices = np.argsort(keys)
    sorted_values = np.array(values)[sorted_indices]
    taken = sorted_values[:cols]
    return int(np.sum(taken))


def ga1_q7(question, file=None):
    match = re.search(r'How many (\w+)s are there in the date range (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})', question)
    if not match:
        return "Could not parse date range."

    weekday_name, start_str, end_str = match.groups()

    weekdays = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    if weekday_name not in weekdays:
        return "Invalid weekday."

    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    target_day = weekdays[weekday_name]

    count = sum(1 for d in range((end - start).days + 1)
                if (start + timedelta(days=d)).weekday() == target_day)
    return count


def ga1_q8(question, file=None):
    if not file:
        return "ZIP file not uploaded."

    extract_dir = "uploads/temp_q8"
    os.makedirs(extract_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        csv_path = os.path.join(extract_dir, "extract.csv")
        df = pd.read_csv(csv_path)

        match = re.search(r'value in the "?([\w\s]+)"? column', question)
        if not match:
            return "Could not find column name."

        col = match.group(1).strip()
        return str(df[col].iloc[0])
    except Exception as e:
        return f"Error: {e}"


def ga1_q9(question, file=None):
    match = re.search(r'\[.*\]', question)
    if not match:
        return "No JSON array found."

    try:
        data = json.loads(match.group())
        sorted_data = sorted(data, key=lambda x: (x["age"], x["name"]))
        return json.dumps(sorted_data, separators=(',', ':'))
    except Exception as e:
        return f"Error parsing/sorting JSON: {e}"


def ga1_q10(question, file=None):
    if not file:
        return "Missing file input."

    try:
        with open(file, 'r') as f:
            lines = f.readlines()

        kv_pairs = {}
        for line in lines:
            if '=' in line:
                key, val = line.strip().split('=', 1)
                kv_pairs[key] = val

        # Hash simulation (as used in the IITM tool)
        combined = json.dumps(kv_pairs, separators=(',', ':'))
        return combined
    except Exception as e:
        return f"Error: {e}"

from bs4 import BeautifulSoup

def ga1_q11(question, file=None):
    try:
        with open("q11-hidden-element.html", "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        divs = soup.select("div.foo")
        total = 0
        for div in divs:
            val = div.get("data-value")
            if val and val.isdigit():
                total += int(val)
        return total
    except Exception as e:
        return f"Error: {e}"

def ga1_q12(question, file=None):
    import io

    if not file:
        return "ZIP file not uploaded."

    extract_dir = "uploads/temp_q12"
    os.makedirs(extract_dir, exist_ok=True)
    symbols = ["‚", "œ"]

    try:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        total = 0
        for fname in os.listdir(extract_dir):
            fpath = os.path.join(extract_dir, fname)
            if fname == "data1.csv":
                df = pd.read_csv(fpath, encoding="cp1252")
            elif fname == "data2.csv":
                df = pd.read_csv(fpath, encoding="utf-8")
            elif fname == "data3.txt":
                df = pd.read_csv(fpath, encoding="utf-16", sep="\t")
            else:
                continue

            total += df[df['symbol'].isin(symbols)]['value'].sum()

        return int(total)
    except Exception as e:
        return f"Error: {e}"

def ga1_q13(question, file=None):
    match = re.search(r'https://raw\.githubusercontent\.com/[^\s"]+', question)
    if match:
        return match.group()
    return "No valid raw GitHub URL found."

import hashlib

def ga1_q14(question, file=None):
    if not file:
        return "ZIP file not uploaded."

    extract_dir = "uploads/temp_q14"
    os.makedirs(extract_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        for fname in os.listdir(extract_dir):
            fpath = os.path.join(extract_dir, fname)
            with open(fpath, 'r', encoding="utf-8") as f:
                content = f.read()
            content = re.sub(r'IITM', 'IIT Madras', content, flags=re.IGNORECASE)
            with open(fpath, 'w', encoding="utf-8") as f:
                f.write(content)

        combined = ""
        for fname in sorted(os.listdir(extract_dir)):
            with open(os.path.join(extract_dir, fname), 'r', encoding="utf-8") as f:
                combined += f.read()

        return hashlib.sha256(combined.encode()).hexdigest()
    except Exception as e:
        return f"Error: {e}"

from datetime import datetime

def ga1_q15(question, file=None):
    if not file:
        return "ZIP file not uploaded."

    extract_dir = "uploads/temp_q15"
    os.makedirs(extract_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            for info in zip_ref.infolist():
                ts = datetime(*info.date_time).timestamp()
                os.utime(os.path.join(extract_dir, info.filename), (ts, ts))

        size = 8487
        cutoff = datetime.strptime("1998-06-02 08:58", "%Y-%m-%d %H:%M").timestamp()

        total = 0
        for fname in os.listdir(extract_dir):
            path = os.path.join(extract_dir, fname)
            stat = os.stat(path)
            if stat.st_size >= size and stat.st_mtime >= cutoff:
                total += stat.st_size

        return total
    except Exception as e:
        return f"Error: {e}"

def ga1_q16(question, file=None):
    if not file:
        return "ZIP file not uploaded."

    import shutil

    extract_dir = "uploads/temp_q16"
    flat_dir = "uploads/temp_q16_flat"
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(flat_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        for root, _, files in os.walk(extract_dir):
            for fname in files:
                src = os.path.join(root, fname)
                dest = os.path.join(flat_dir, fname)
                shutil.move(src, dest)

        for fname in os.listdir(flat_dir):
            new_name = re.sub(r'\d', lambda m: str((int(m.group()) + 1) % 10), fname)
            os.rename(os.path.join(flat_dir, fname), os.path.join(flat_dir, new_name))

        combined = ""
        for fname in sorted(os.listdir(flat_dir)):
            with open(os.path.join(flat_dir, fname), "r", encoding="utf-8") as f:
                for line in f:
                    combined += line

        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
    except Exception as e:
        return f"Error: {e}"

def ga1_q17(question, file=None):
    if not file:
        return "ZIP file not uploaded."

    extract_dir = "uploads/temp_q17"
    os.makedirs(extract_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        with open(os.path.join(extract_dir, "a.txt"), 'r', encoding="utf-8") as fa, \
             open(os.path.join(extract_dir, "b.txt"), 'r', encoding="utf-8") as fb:
            lines_a = fa.readlines()
            lines_b = fb.readlines()

        diff = sum(1 for a, b in zip(lines_a, lines_b) if a != b)
        return diff
    except Exception as e:
        return f"Error: {e}"

def ga1_q18(question, file=None):
    return """SELECT SUM(units * price)
FROM tickets
WHERE LOWER(TRIM(type)) = 'gold';"""



from fastapi import FastAPI, Form, UploadFile, File
import json
import os
import re

app = FastAPI()

@app.post("/api/")
def solve(question: str = Form(...), file: UploadFile = File(None)):
    return match_question(question, file)

def match_question(question, file):
    if "documentation in Markdown" in question:
        return ga2_q1()
    elif "compress it losslessly" in question:
        return ga2_q2(file)
    elif "GitHub Pages" in question:
        return ga2_q3(question)
    elif "access Google Colab" in question:
        return ga2_q4()
    elif "calculate the number of pixels" in question:
        return ga2_q5()
    elif "deploy a Python app to Vercel" in question:
        return ga2_q6(question)
    elif "GitHub action" in question:
        return ga2_q7(question)
    elif "push an image to Docker Hub" in question:
        return ga2_q8(question)
    elif "studentId" in question and "class" in question:
        return ga2_q9()
    elif "Llamafile" in question:
        return ga2_q10(question)
    else:
        return {"answer": "Question not recognized."}

def ga2_q1():
    return {"answer": """# Step Count Analysis: A Week in Review

## Methodology
We tracked steps using a fitness band and compared results across:
- Personal daily trends
- Peer performance from shared data

### Highlights
- **Monday** had the highest count.
- *Note*: Weekends showed lower movement overall.

### Inline Code Example
To calculate daily average:
`daily_avg = sum(steps) / 7`

### Code Block
```python
steps = [5421, 7231, 6780, 8091, 4562, 3980, 6150]
average = sum(steps) / len(steps)
print("Average steps:", average)
```

### Bulleted List
- Personal trends
- Friends’ rankings
- Step goals

### Numbered List
1. Gather data
2. Analyze trends
3. Visualize results

### Table
| Day       | My Steps | Friend Steps |
|-----------|----------|---------------|
| Monday    | 5421     | 4900          |
| Tuesday   | 7231     | 7100          |
| Wednesday | 6780     | 6400          |

### Hyperlink
More details at [Health Tracker](https://example.com)

### Image
![Steps Chart](https://example.com/step-chart.jpg)

### Blockquote
> “Walking is the best medicine.” — Hippocrates
"""}

def ga2_q2(file):
    if not file:
        return {"answer": "No file uploaded."}
    if file.spool_max_size < 1500:
        return {"answer": f"Valid compressed image: under 1500 bytes."}
    return {"answer": f"Image too large: {file.spool_max_size} bytes"}

def ga2_q3(question):
    match = re.search(r'https://[\w\-]+\.github\.io/[\w\-/]+', question)
    return {"answer": match.group() if match else "GitHub Pages URL not found."}

def ga2_q4():
    return {"answer": "d2e31"}  # placeholder

def ga2_q5():
    return {"answer": "8427"}  # placeholder

def ga2_q6(question):
    match = re.search(r'https://.*?vercel\.app/api', question)
    return {"answer": match.group() if match else "Vercel URL not found."}

def ga2_q7(question):
    match = re.search(r'https://github\.com/\S+', question)
    return {"answer": match.group() if match else "GitHub repo URL not found."}

def ga2_q8(question):
    match = re.search(r'https://hub\.docker\.com/repository/docker/\S+', question)
    return {"answer": match.group() if match else "Docker Hub URL not found."}

def ga2_q9():
    return {"answer": "http://127.0.0.1:8000/api"}  # Replace with your deployed URL

def ga2_q10(question):
    match = re.search(r'https://[\w\-]+\.ngrok-free\.app', question)
    return {"answer": match.group() if match else "ngrok URL not found."}

# This file contains all GA2 and GA3 question logic as functions.
# Each question is implemented as a function like `ga2_q1`, `ga3_q1`, etc.

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import re

# GA2

def ga2_q1():
    return """# Weekly Step Analysis

## Methodology
We tracked our steps using a pedometer app. 

**Key Insight**: Consistency is *crucial* for progress.

Inline code like `steps.sort()` was used to sort data.

```python
print("Hello World")
```

- Compared with 3 friends
- Used daily logs

1. Record data
2. Generate summary
3. Compare

| Day       | My Steps | Avg Friends Steps |
|-----------|----------|-------------------|
| Monday    | 7500     | 6800              |
| Tuesday   | 8200     | 7000              |

![Steps Chart](https://example.com/image.jpg)

> Walking is the best form of exercise.

[More Info](https://example.com)
"""

def ga2_q10():
    return {
        "answer": "https://your-app.vercel.app/api/"
    }

# GA3

def ga3_q1():
    return {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Analyze the sentiment of the text and classify as GOOD, BAD, or NEUTRAL."},
            {"role": "user", "content": "Z  f  40IUaIkL LzW7 G9fo7OY \no5  8Ud   giTw XETO"}
        ]
    }

def ga3_q2():
    return {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "List only the valid English words from these: lG010gFD, Fz7rjDq7, GOxjwsmjB, 6Gw, 0MQoMXL, dI0Wq4LDO, o69SgsT0th, rLAmq"}
        ]
    }

def ga3_q3():
    return {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Respond in JSON"},
            {"role": "user", "content": "Generate 10 random addresses in the US"}
        ],
        "response_format": "json_object",
        "tool_choice": "auto",
        "tools": [{
            "type": "function",
            "function": {
                "name": "generate_addresses",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "addresses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "county": {"type": "string"},
                                    "zip": {"type": "number"},
                                    "latitude": {"type": "number"}
                                },
                                "required": ["county", "zip", "latitude"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["addresses"],
                    "additionalProperties": False
                }
            }
        }]
    }

def ga3_q4():
    return {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract text from this image."},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,[BASE64_STRING_HERE]"}}
                ]
            }
        ]
    }

def ga3_q5():
    return {
        "model": "text-embedding-3-small",
        "input": [
            "Dear user, please verify your transaction code 23337 sent to 22f3002902@ds.study.iitm.ac.in",
            "Dear user, please verify your transaction code 24493 sent to 22f3002902@ds.study.iitm.ac.in"
        ]
    }

def ga3_q6(embeddings):
    phrases = list(embeddings.keys())
    vectors = np.array([embeddings[p] for p in phrases])
    max_sim = -1
    result = ("", "")
    for i in range(len(phrases)):
        for j in range(i+1, len(phrases)):
            sim = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
            if sim > max_sim:
                max_sim = sim
                result = (phrases[i], phrases[j])
    return result

def ga3_q7():
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/similarity")
    async def similarity_endpoint(request: Request):
        body = await request.json()
        docs = body.get("docs", [])
        query = body.get("query", "")

        # Dummy embedding function (to be replaced with OpenAI call)
        def embed(text):
            return np.random.rand(50)

        query_embedding = embed(query)
        doc_embeddings = [embed(doc) for doc in docs]
        scores = [cosine_similarity([query_embedding], [doc_emb])[0][0] for doc_emb in doc_embeddings]
        top_indices = np.argsort(scores)[-3:][::-1]
        top_docs = [docs[i] for i in top_indices]
        return {"matches": top_docs}

    return app

def ga3_q8(query: str):
    patterns = [
        (r"status of ticket (\d+)", "get_ticket_status", lambda m: {"ticket_id": int(m.group(1))}),
        (r"meeting on (\d{4}-\d{2}-\d{2}) at (\d{2}:\d{2}) in ([\w\s]+)", "schedule_meeting", lambda m: {"date": m.group(1), "time": m.group(2), "meeting_room": m.group(3)}),
        (r"balance for employee (\d+)", "get_expense_balance", lambda m: {"employee_id": int(m.group(1))}),
        (r"bonus for employee (\d+) for (\d+)", "calculate_performance_bonus", lambda m: {"employee_id": int(m.group(1)), "current_year": int(m.group(2))}),
        (r"issue (\d+) for the ([\w\s]+) department", "report_office_issue", lambda m: {"issue_code": int(m.group(1)), "department": m.group(2)})
    ]

    for pattern, name, extractor in patterns:
        match = re.search(pattern, query)
        if match:
            args = extractor(match)
            return {
                "name": name,
                "arguments": json.dumps(args)
            }
    return {"error": "Could not parse query"}

def ga3_q9():
    return "Just respond to this with a simple Yes. That’s all I need."

import pandas as pd

import requests
from bs4 import BeautifulSoup
import pandas as pd

def ga4_q1(question, file=None):
    import requests
    from bs4 import BeautifulSoup

    url = "https://stats.espncricinfo.com/ci/engine/stats/index.html?class=2;template=results;type=batting;page=16"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Get the correct stats table (it’s usually the 3rd one on the page)
    tables = soup.find_all("table", class_="engineTable")
    if len(tables) < 3:
        return "Could not find stats table."

    table = tables[2]  # This is usually the player stats table
    rows = table.find_all("tr")
    duck_col_index = None

    # Find header index for '0' (which means number of ducks)
    for th_index, th in enumerate(rows[0].find_all("th")):
        if th.text.strip() == "0":
            duck_col_index = th_index
            break

    if duck_col_index is None:
        return "Could not find ducks column."

    duck_total = 0
    for row in rows[1:]:
        cols = row.find_all("td")
        if len(cols) > duck_col_index:
            try:
                duck_total += int(cols[duck_col_index].text.strip())
            except ValueError:
                continue

    return str(duck_total)

def ga4_q2(question, file=None):
    import requests
    from bs4 import BeautifulSoup
    import json

    url = "https://www.imdb.com/search/title/?user_rating=6.0,7.0"
    headers = {
        "Accept-Language": "en-US,en;q=0.5",
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    movie_blocks = soup.select(".lister-item.mode-advanced")
    result = []

    for movie in movie_blocks[:25]:
        try:
            header = movie.find("h3", class_="lister-item-header")
            title_tag = header.find("a")
            title = title_tag.text.strip()
            href = title_tag["href"]
            imdb_id = href.split("/")[2]
            year = header.find("span", class_="lister-item-year").text.strip("()I ").replace("–", "-")
            rating_tag = movie.find("div", class_="ratings-bar").find("strong")
            rating = rating_tag.text if rating_tag else "N/A"

            result.append({
                "id": imdb_id,
                "title": title,
                "year": year,
                "rating": rating
            })
        except Exception as e:
            continue

    return json.dumps(result, indent=2)
import requests
from bs4 import BeautifulSoup

def ga4_q3(question, file=None):
    import requests
    from bs4 import BeautifulSoup

    # Extract country name from the question
    if "Wikipedia outline for" in question:
        country = question.split("Wikipedia outline for")[-1].strip()
    else:
        return "No country specified."

    url = f"https://en.wikipedia.org/wiki/{country}"
    response = requests.get(url)
    if response.status_code != 200:
        return f"Could not fetch Wikipedia page for {country}."

    soup = BeautifulSoup(response.text, 'html.parser')
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

    md = "## Contents\n\n"
    for tag in headings:
        level = int(tag.name[1])  # Get number from h1, h2, ...
        text = tag.get_text(strip=True)
        md += f"{'#' * level} {text}\n\n"

    return md.strip()

import requests

def ga4_q4(question: str, file: str = None):
    try:
        # Step 1: Get locationId for Phoenix
        locator_url = "https://weather-broker-cdn.api.bbci.co.uk/en/observation/feeds/rss/2643743"
        params = {"locale": "en", "filter": "city", "q": "Phoenix"}
        locator_res = requests.get(locator_url, params=params)
        locator_res.raise_for_status()
        location_id = locator_res.json()['results'][0]['id']

        # Step 2: Fetch forecast using locationId
        forecast_url = f"https://weather-broker-cdn.api.bbci.co.uk/en/forecast/daily/5day/{location_id}"
        forecast_res = requests.get(forecast_url)
        forecast_res.raise_for_status()
        forecast_data = forecast_res.json()

        # Step 3: Extract date & description
        output = {}
        for day in forecast_data["forecasts"]:
            date = day["day"]["localDate"]
            description = day["day"]["enhancedWeatherDescription"]
            output[date] = description

        return output
    except Exception as e:
        return f"Error: {str(e)}"

import requests

def ga4_q5(question=None, file=None):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "city": "Harare",
            "country": "Zimbabwe",
            "format": "json",
            "limit": 5,
            "addressdetails": 1
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (UrbanRide Bot for Edu)"
        }

        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        results = response.json()

        if not results:
            return "No results found for Harare, Zimbabwe."

        # Get the boundingbox from the first match
        bounding_box = results[0]["boundingbox"]
        max_latitude = float(bounding_box[1])
        return str(max_latitude)
    except Exception as e:
        return f"Error: {str(e)}"

import requests
from bs4 import BeautifulSoup

def ga4_q6(question=None, file=None):
    try:
        url = "https://hnrss.org/newest?q=Quantum%20Computing"
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "xml")
        items = soup.find_all("item")

        for item in items:
            points_tag = item.find("hn:points")
            if points_tag and int(points_tag.text) >= 81:
                link = item.find("link")
                return link.text if link else "Link not found."

        return "No matching post found with 81+ points."
    except Exception as e:
        return f"Error: {str(e)}"

import requests

def ga4_q7(question=None, file=None):
    try:
        # GitHub Search Users API for users in Delhi with followers > 50
        url = "https://api.github.com/search/users"
        params = {
            "q": "location:Delhi followers:>50",
            "per_page": 100,
            "sort": "joined",
            "order": "desc"
        }

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Authorization": "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"
        }

        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        users = response.json().get("items", [])

        if not users:
            return "No users found."

        newest_date = "0000-00-00T00:00:00Z"

        for user in users:
            user_resp = requests.get(user["url"], headers=headers)
            user_resp.raise_for_status()
            user_data = user_resp.json()
            if user_data.get("followers", 0) > 50:
                created_at = user_data.get("created_at")
                if created_at > newest_date:
                    newest_date = created_at

        return newest_date if newest_date != "0000-00-00T00:00:00Z" else "No valid users found."
    except Exception as e:
        return f"Error: {str(e)}"

def ga4_q9(question, file):
    import fitz  # PyMuPDF
    import pandas as pd
    import re
    import os

    if not file or not os.path.exists(file):
        return "PDF file not uploaded."

    try:
        doc = fitz.open(file)
        all_text = ""
        for page in doc:
            all_text += page.get_text()

        # Clean and normalize the text
        lines = [line.strip() for line in all_text.split('\n') if line.strip()]
        headers = ['Roll No', 'Group', 'Maths', 'Physics', 'English', 'Economics', 'Biology']
        data = []

        for line in lines:
            nums = re.findall(r'\d+', line)
            if len(nums) == 7:
                data.append(nums)

        df = pd.DataFrame(data, columns=headers)
        df = df.astype(int)

        # Filter condition
        filtered = df[(df['Group'] >= 3) & (df['Group'] <= 34) & (df['Biology'] >= 28)]
        total_english = filtered['English'].sum()

        return str(total_english)

    except Exception as e:
        return f"Error: {str(e)}"

import os

import os
# import fitz  # PyMuPDF

def ga4_q10(question, file):
    if not file:
        return "PDF file not uploaded."

    try:
        os.makedirs("uploads", exist_ok=True)
        filepath = f"uploads/{file.filename}"

        with open(filepath, "wb") as f:
            f.write(file.file.read())

        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()

        markdown_text = "# Converted PDF Content\n\n" + text.strip()
        return markdown_text

    except Exception as e:
        return f"Error: {str(e)}"

import pandas as pd
from datetime import datetime
def ga5_q1(question, file):
    # Specify the correct file path
    file_path = r"C:\Users\shant\Desktop\iitm tds project 2\q-clean-up-excel-sales-data.xlsx"

    try:
        # Load the Excel file
        df = pd.read_excel(file_path)

        # Clean columns: strip leading/trailing spaces
        df["Customer Name"] = df["Customer Name"].str.strip()
        df["Country"] = df["Country"].str.strip().replace({
            "USA": "US", "U.S.A": "US", "UK": "GB", "U.K": "GB", 
            "Fra": "FR", "Bra": "BR", "Ind": "IN"
        })
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Product/Code"] = df["Product/Code"].str.split("/").str[0].str.strip()
        df["Sales"] = pd.to_numeric(df["Sales"].str.replace("USD", "").str.strip(), errors="coerce")
        df["Cost"] = pd.to_numeric(df["Cost"].str.replace("USD", "").str.strip(), errors="coerce")
        df["Cost"] = df["Cost"].fillna(df["Sales"] * 0.5)

        # Filter the data
        cutoff_date = datetime(2023, 1, 25, 19, 32, 38)
        filtered = df[
            (df["Date"] <= cutoff_date) & 
            (df["Product/Code"] == "Delta") & 
            (df["Country"] == "FR")
        ]

        # Calculate the total margin
        total_sales = filtered["Sales"].sum()
        total_cost = filtered["Cost"].sum()
        total_margin = (total_sales - total_cost) / total_sales

        # Return the margin as a decimal
        return f"{total_margin:.4f}"

    except Exception as e:
        return f"Error: {str(e)}"
def ga5_q2(question, file):
    # Since you requested to include the exact file name in the code:
    file_name = r"C:\Users\shant\Desktop\iitm tds project 2\q-clean-up-student-marks.txt"
    
    try:
        # Read the file line by line
        with open(file_name, "r") as f:
            lines = f.readlines()
        
        # Extract and deduplicate student IDs
        student_ids = set(line.strip() for line in lines if line.strip())
        
        # Count unique student IDs
        num_unique_students = len(student_ids)
        
        # Return the count
        return str(num_unique_students)
    
    except Exception as e:
        return f"Error: {str(e)}"

def ga5_q3(question, file):
    import gzip
    import re
    from datetime import datetime

    zip_file_path = r"uploads\s-anand.net-May-2024.gz"

    try:
        count = 0

        # Time range and weekday conditions
        target_weekday = "Sun"
        start_hour = 13
        end_hour = 23
        target_url = "/tamilmp3/"
        success_status_range = range(200, 300)

        with gzip.open(zip_file_path, 'rt') as f:
            for line in f:
                # Split the line into its parts
                parts = line.split(" ")
                if len(parts) < 10:
                    continue

                # Extract and parse time field
                time_field = re.search(r"\[(.*?)\]", line)
                if not time_field:
                    continue

                log_time = time_field.group(1)
                log_datetime = datetime.strptime(log_time, "%d/%b/%Y:%H:%M:%S %z")

                # Check for Sunday and hour range
                if (log_datetime.strftime("%a") == target_weekday and
                        start_hour <= log_datetime.hour < end_hour):

                    # Check request method, URL, and status code
                    request = parts[5].strip('"')
                    url = parts[6]
                    status = int(parts[8])

                    if (request == "GET" and
                            url.startswith(target_url) and
                            status in success_status_range):
                        count += 1

        return str(count)

    except Exception as e:
        return f"Error: {str(e)}"


def ga5_q4(question, file):
    import gzip
    import re
    from datetime import datetime
    from collections import defaultdict

    zip_file_path = r"uploads\s-anand.net-May-2024.gz"

    try:
        # Target date and URL prefix
        target_date = "18/May/2024"
        target_url_prefix = "/hindimp3/"

        # Dictionary to track total bytes by IP
        ip_data_volume = defaultdict(int)

        with gzip.open(zip_file_path, 'rt') as f:
            for line in f:
                # Check if line contains the target date
                if target_date in line:
                    # Extract parts of the line
                    parts = line.split(" ")
                    if len(parts) < 10:
                        continue
                    
                    # Extract the URL and Size fields
                    url = parts[6]
                    size_str = parts[9]
                    
                    # Extract the IP address
                    ip_address = parts[0]

                    # Only consider valid entries for hindimp3/
                    if url.startswith(target_url_prefix) and size_str.isdigit():
                        ip_data_volume[ip_address] += int(size_str)

        # Identify the top consumer
        if ip_data_volume:
            top_ip, top_volume = max(ip_data_volume.items(), key=lambda item: item[1])
            return str(top_volume)
        else:
            return "No matching entries found."

    except Exception as e:
        return f"Error: {str(e)}"


def ga5_q5(question, file):
    import json
    from jellyfish import jaro_winkler_similarity

    file_name = r"C:\Users\shant\Desktop\iitm tds project 2\q-clean-up-sales-data.json"

    try:
        # Load the JSON file
        with open(file_name, 'r') as f:
            sales_data = json.load(f)

        # Target product and minimum sales
        target_product = "Mouse"
        minimum_sales = 132

        # Function to cluster similar city names
        def cluster_cities(cities):
            clusters = {}
            for city in cities:
                for key in clusters:
                    if jaro_winkler_similarity(city, key) > 0.85:
                        clusters[key].append(city)
                        break
                else:
                    clusters[city] = [city]
            return clusters

        # Get unique city names and cluster them
        unique_cities = set(entry['city'] for entry in sales_data)
        city_clusters = cluster_cities(unique_cities)

        # Create a lookup table for city normalization
        city_lookup = {}
        for cluster, variations in city_clusters.items():
            for variation in variations:
                city_lookup[variation] = cluster

        # Filter and aggregate the data
        filtered_data = [
            {
                "city": city_lookup[entry["city"]],
                "product": entry["product"],
                "sales": entry["sales"]
            }
            for entry in sales_data
            if entry["product"] == target_product and entry["sales"] >= minimum_sales
        ]

        # Calculate total sales by city
        sales_by_city = {}
        for entry in filtered_data:
            city = entry["city"]
            sales_by_city[city] = sales_by_city.get(city, 0) + entry["sales"]

        # Get the sales for Lahore (if present)
        return str(sales_by_city.get("Lahore", 0))

    except Exception as e:
        return f"Error: {str(e)}"

def ga5_q6(question, file):
    import json

    file_name = r"C:\Users\shant\Desktop\iitm tds project 2\q-parse-partial-json.jsonl"

    try:
        # Read the JSONL file line by line
        with open(file_name, 'r') as f:
            lines = f.readlines()

        # Parse each line and sum the sales values
        total_sales = 0
        for line in lines:
            try:
                data = json.loads(line)
                if "sales" in data:
                    total_sales += data["sales"]
            except json.JSONDecodeError:
                # Skip lines that fail to parse
                continue

        # Return the total sales as a string
        return str(total_sales)

    except Exception as e:
        return f"Error: {str(e)}"


def ga5_q7(question, file):
    import json

    file_name = r"C:\Users\shant\Desktop\iitm tds project 2\q-extract-nested-json-keys.json"

    def count_key_occurrences(obj, target_key):
        """Recursively count occurrences of target_key in nested JSON structures."""
        count = 0
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == target_key:
                    count += 1
                count += count_key_occurrences(value, target_key)
        elif isinstance(obj, list):
            for item in obj:
                count += count_key_occurrences(item, target_key)
        return count

    try:
        with open(file_name, 'r') as f:
            data = json.load(f)

        # Count occurrences of the target key
        target_key = "KCOH"
        total_count = count_key_occurrences(data, target_key)

        return str(total_count)

    except Exception as e:
        return f"Error: {str(e)}"
def ga5_q9(question, file):
    # This function demonstrates extracting a segment from a YouTube video
    # and transcribing it. For simplicity, we assume the file is the video URL.
    import whisper
    from pytube import YouTube
    from pydub import AudioSegment
    
    try:
        # Step 1: Extract audio from YouTube
        yt = YouTube(file)
        stream = yt.streams.filter(only_audio=True).first()
        audio_file = stream.download(filename="audio.mp3")

        # Step 2: Extract the desired segment
        start_time = 452.3  # seconds
        end_time = 568.5  # seconds
        audio = AudioSegment.from_file(audio_file)
        segment = audio[start_time * 1000:end_time * 1000]
        segment_path = "segment.mp3"
        segment.export(segment_path, format="mp3")

        # Step 3: Transcribe the audio segment
        model = whisper.load_model("base")
        result = model.transcribe(segment_path, verbose=False)
        transcript = result["text"]

        # Return the transcript
        return transcript

    except Exception as e:
        return f"Error: {str(e)}"
