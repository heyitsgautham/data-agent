import re
@app.post("/aianalyst/")
async def aianalyst(
    file: UploadFile = File(...),
    image: UploadFile = File(None),
    csv: UploadFile = File(None)
):
    time_start = time.time()
    content = await file.read()
    question_text = content.decode("utf-8")

    # Handle image if provided (existing logic)
    if image:
        try:
            image_bytes = await image.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            if not ocr_api_key:
                print("⚠️ OCR_API_KEY not found - skipping image processing")
                question_text += "\n\nOCR API key not configured - image text extraction skipped"
            else:
                async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                    form_data = {
                        "base64Image": f"data:image/png;base64,{base64_image}",
                        "apikey": ocr_api_key,
                        "language": "eng",
                        "scale": "true",
                        "OCREngine": "1"
                    }
                    
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                    
                    response = await client.post(OCR_API_URL, data=form_data, headers=headers)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if not result.get('IsErroredOnProcessing', True):
                            parsed_results = result.get('ParsedResults', [])
                            if parsed_results:
                                image_text = parsed_results[0].get('ParsedText', '').strip()
                                if image_text:
                                    question_text += f"\n\nExtracted from image:\n{image_text}"
                                    print("✅ Text extracted from image")
                    else:
                        print(f"❌ OCR API error: {response.status_code}")
                    
        except Exception as e:
            print(f"❌ Error extracting text from image: {e}")

    # Step 3: Handle provided CSV file
    provided_csv_info = None
    if csv:
        try:
            csv_content = await csv.read()
            csv_df = pd.read_csv(StringIO(csv_content.decode("utf-8")))
            
            # Clean the CSV
            sourcer = data_scrape.ImprovedWebScraper()
            cleaned_df, formatting_results = await sourcer.numeric_formatter.format_dataframe_numerics(csv_df)
            
            # Save as ProvidedCSV.csv
            cleaned_df.to_csv("ProvidedCSV.csv", index=False, encoding="utf-8")
            
            provided_csv_info = {
                "filename": "ProvidedCSV.csv",
                "shape": cleaned_df.shape,
                "columns": list(cleaned_df.columns),
                "sample_data": cleaned_df.head(3).to_dict('records'),
                "description": "User-provided CSV file (cleaned and formatted)",
                "formatting_applied": formatting_results
            }
            
            print(f"📝 Provided CSV processed: {cleaned_df.shape} rows, saved as ProvidedCSV.csv")
            
        except Exception as e:
            print(f"❌ Error processing provided CSV: {e}")

    # Step 4: Extract all URLs and database files from question
    print("🔍 Extracting all data sources from question...")
    extracted_sources = await extract_all_urls_and_databases(question_text)
    
    print(f"📊 Found {len(extracted_sources.get('scrape_urls', []))} URLs to scrape")
    print(f"📊 Found {len(extracted_sources.get('database_files', []))} database files")

    # Step 5: Scrape all URLs and save as CSV files
    scraped_data = []
    if extracted_sources.get('scrape_urls'):
        scraped_data = await scrape_all_urls(extracted_sources['scrape_urls'])

    # Step 6: Get database schemas and sample data
    database_info = []
    if extracted_sources.get('database_files'):
        database_info = await get_database_schemas(extracted_sources['database_files'])

    # Step 7: Create comprehensive data summary
    data_summary = create_data_summary(scraped_data, provided_csv_info, database_info)
    
    # Save data summary for debugging
    with open("data_summary.json", "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(data_summary), f, indent=2)

    print(f"📋 Data Summary: {data_summary['total_sources']} total sources")

    # Utility to safely extract Gemini text
    def extract_gemini_text(response: dict) -> str:
        """
        Safely extract the first text part from a Gemini API response.
        Returns an empty string if the structure is unexpected.
        """
        try:
            candidates = response.get("candidates", [])
            if not candidates:
                return ""
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                return ""
            text = parts[0].get("text", "")
            return text if isinstance(text, str) else ""
        except Exception as e:
            # logger.warning(f"Error extracting Gemini text: {e} | Response: {response}")
            print(f"Warning: Error extracting Gemini text: {e} | Response: {response}")
            return ""

    # Break down tasks
    task_breaker_instructions = read_prompt_file("prompts/task_breaker.txt")
    gemini_response = await ping_gemini(question_text, task_breaker_instructions)
    task_breaked = extract_gemini_text(gemini_response)

    with open("broken_down_tasks.txt", "w", encoding="utf-8") as f:
        f.write(str(task_breaked))

    # Step 8: Generate final code based on all data sources
    # Use unified instructions that handle all source types
    code_instructions = read_prompt_file("prompts/unified_code_instructions.txt")

    context = (
        "ORIGINAL QUESTION: " + question_text + "\n\n" +
        "TASK BREAKDOWN: " + task_breaked + "\n\n" +
        "INSTRUCTIONS: " + code_instructions + "\n\n" +
        "DATA SUMMARY: " + json.dumps(make_json_serializable(data_summary), indent=2)
    )

    # Build explicit allowed files list to prevent model hallucinating file paths
    allowed_paths = []
    if provided_csv_info:
        allowed_paths.append(provided_csv_info.get('filename'))
    for s in scraped_data:
        if 'filename' in s:
            allowed_paths.append(s['filename'])
    for db in database_info:
        if 'source_url' in db:
            allowed_paths.append(db['source_url'])
    # Deduplicate and format
    allowed_paths = list(dict.fromkeys([p for p in allowed_paths if p]))
    allowed_files_text = "ALLOWED_DATA_SOURCES:\n" + "\n".join(allowed_paths) if allowed_paths else "ALLOWED_DATA_SOURCES: NONE"

    # Append allowed files to the LLM context and instruct the model to not access any other files
    context += "\n\n" + "IMPORTANT: You may only read from the following data sources. Do NOT read or write any other file paths.\n" + allowed_files_text
    context += "\n\nIMPORTANT: Do NOT include any comments in the code output. Provide only pure Python code without any inline or block comments."

    # Add explicit instruction to the Horizon system message
    horizon_system_message = (
        "You are a great Python code developer. Who write final code for the answer and our workflow using all the detail provided to you"
        " IMPORTANT: Output only valid Python code without any comments, explanations, markdown formatting, or triple backticks."
    )
    horizon_response = await ping_horizon(context, horizon_system_message)

    # Directly extract raw_code from the response, with fallback
    if "candidates" in horizon_response:
        raw_code = horizon_response["candidates"][0]["content"]["parts"][0]["text"]
    elif "choices" in horizon_response:  # fallback for OpenAI/OpenRouter format
        raw_code = horizon_response["choices"][0]["message"]["content"]
    else:
        raise ValueError(f"Unexpected Horizon response format: {horizon_response}")

    raw_code = raw_code.strip()
    # Remove triple backticks if present, as a fallback
    if "```" in raw_code:
        raw_code = raw_code.replace("```python", "").replace("```", "")
    cleaned_code = raw_code

    with open("chatgpt_code.py", "w") as f:
        f.write(cleaned_code)
    # Remove any 'quality=' parameter from plt.savefig or fig.savefig calls
    try:
        with open("chatgpt_code.py", "r", encoding="utf-8") as _f:
            _code = _f.read()
        # Remove ', quality=...' from savefig calls (e.g., plt.savefig(..., quality=95))
        _code = re.sub(r'(savefig\s*\([^)]*?),\s*quality\s*=\s*[^,)]+', r'\1', _code)
        with open("chatgpt_code.py", "w", encoding="utf-8") as _f:
            _f.write(_code)
    except Exception as _e:
        print(f"Warning: failed to clean 'quality=' from savefig: {_e}")

    # --- Sanitize generated code: block or replace disallowed file accesses ---
    try:
        with open("chatgpt_code.py", "r", encoding="utf-8") as _f:
            _code = _f.read()
        _modified = False
        # Patterns to check: pd.read_csv('...'), pd.read_parquet('...'), read_csv_auto('...'), read_parquet('...'), open('...')
        patterns = [
            (r"pd\.read_csv\([\'\"]([^\'\"]+)[\'\"]", 'csv'),
            (r"pd\.read_parquet\([\'\"]([^\'\"]+)[\'\"]", 'parquet'),
            (r"read_csv_auto\([\'\"]([^\'\"]+)[\'\"]", 'csv'),
            (r"read_parquet\([\'\"]([^\'\"]+)[\'\"]", 'parquet'),
            (r"open\([\'\"]([^\'\"]+)[\'\"]", 'open')
        ]
        for patt, ptype in patterns:
            for m in re.finditer(patt, _code):
                path = m.group(1)
                # If path is not explicitly allowed, replace or block
                if path not in allowed_paths and os.path.basename(path) not in allowed_paths:
                    _modified = True
                    start = m.start()
                    # Find the start and end of the line containing this match
                    line_start = _code.rfind('\n', 0, start) + 1
                    line_end = _code.find('\n', start)
                    if line_end == -1:
                        line_end = len(_code)
                    offending_line = _code[line_start:line_end]
                    # Check if the offending line assigns a variable (contains '=' before the pattern)
                    eq_pos = offending_line.find('=')
                    patt_pos = offending_line.find(m.group(0))
                    if eq_pos != -1 and eq_pos < patt_pos:
                        # Replace only the right-hand side with ''
                        # e.g., base_path = pd.read_csv('notallowed.csv')  => base_path = ''
                        var_name = offending_line[:eq_pos+1]  # include '='
                        replacement = var_name + " ''"
                        # preserve indentation
                        leading_ws = len(offending_line) - len(offending_line.lstrip())
                        replacement = " " * leading_ws + replacement
                        _code = _code[:line_start] + replacement + _code[line_end:]
                    else:
                        # For read_parquet, do NOT replace the path, just leave the original line as is
                        if ptype == 'parquet':
                            continue  # skip replacing for read_parquet
                        # Replace only the offending path inside quotes with an empty string, keep the rest of the line
                        offending_path = path
                        new_line = re.sub(
                            r"(['\"])(%s)\1" % re.escape(offending_path),
                            r"\1\1",
                            offending_line,
                            count=1
                        )
                        # Preserve indentation
                        leading_ws = len(offending_line) - len(offending_line.lstrip())
                        replacement = " " * leading_ws + new_line.lstrip()
                        _code = _code[:line_start] + replacement + _code[line_end:]
        # Remove the logic that forcibly changes duckdb.connect(...) to duckdb.connect()
        if _modified:
            with open("chatgpt_code.py", "w", encoding="utf-8") as _f:
                _f.write(_code)
    except Exception as _e:
        print(f"Warning: failed to sanitize file paths in generated code: {_e}")

    # Execute the code
    try:
        result = subprocess.run(
            ["python", "chatgpt_code.py"],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Check for missing module error and try to install
        missing_module = None
        if result.returncode != 0:
            match = re.search(r"No module named '([^']+)'", result.stderr)
            if match:
                missing_module = match.group(1)
                print(f"⚠️ Detected missing module: {missing_module}. Attempting to install...")
                try:
                    subprocess.run(
                        ["pip", "install", missing_module],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    # Re-run the script after installing the module
                    result = subprocess.run(
                        ["python", "chatgpt_code.py"],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                except Exception as e:
                    print(f"❌ Failed to install missing module {missing_module}: {e}")

        if result.returncode == 0:
            stdout = result.stdout.strip()
            json_output = extract_json_from_output(stdout)
            
            if is_valid_json_output(json_output):
                try:
                    output_data = json.loads(json_output)
                    print("✅ Code executed successfully")
                    return output_data
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {str(e)[:100]}")
            else:
                print(f"Output doesn't look like JSON: {json_output[:100]}")
        else:
            print(f"Execution error: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("Code execution timed out")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # Code fixing attempts (existing logic)
    max_fix_attempts = 3
    fix_attempt = 0
    
    while fix_attempt < max_fix_attempts:
        fix_attempt += 1
        print(f"🔧 Attempting to fix code (attempt {fix_attempt}/{max_fix_attempts})")
        
        try:
            with open("chatgpt_code.py", "r", encoding="utf-8") as code_file:
                code_content = code_file.read()
            
            try:
                result = subprocess.run(
                    ["python", "chatgpt_code.py"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                # Check for missing module error and try to install
                missing_module = None
                if result.returncode != 0:
                    match = re.search(r"No module named '([^']+)'", result.stderr)
                    if match:
                        missing_module = match.group(1)
                        print(f"⚠️ Detected missing module during fix: {missing_module}. Attempting to install...")
                        try:
                            subprocess.run(
                                ["pip", "install", missing_module],
                                capture_output=True,
                                text=True,
                                timeout=60
                            )
                            # Re-run the script after installing the module
                            result = subprocess.run(
                                ["python", "chatgpt_code.py"],
                                capture_output=True,
                                text=True,
                                timeout=120
                            )
                        except Exception as e:
                            print(f"❌ Failed to install missing module {missing_module}: {e}")
                error_context = f"Return code: {result.returncode}\nStderr: {result.stderr}\nStdout: {result.stdout}"
            except Exception as e:
                error_context = f"Execution failed with exception: {str(e)}"
            
            error_message = f"Error: {error_context}\n\nCode:\n{code_content}\n\nTask breakdown:\n{task_breaked}"
            
            fix_prompt = (
                """URGENT CODE FIXING TASK:
                    CURRENT BROKEN CODE:
                    ```python
                    {current_code}
                    ```
                    ERROR DETAILS:
                    {initial_error}
                    AVAILABLE DATA (use these exact sources):
                    {json.dumps(data_summary, indent=2)}

                    ORIGINAL TASK:
                    {question_text}

                    TASK BREAKDOWN:
                    {task_breakdown}
                    FIXING INSTRUCTIONS:
                    1. Do NOT add new logic, data sources, or change the question requirements.
                    2. Only fix the exact errors found. Keep ALL original logic and structure unchanged.
                    3. Do NOT replace real data with fake or random data.
                    4. Do NOT hallucinate functions, variables, or imports.
                    5. Output the corrected code exactly as before, but fixed for the error.
                    1. Fix the specific error mentioned above
                    2. Use ONLY the data sources listed in AVAILABLE DATA section
                    3. DO NOT add placeholder URLs or fake data
                    4. DO NOT create imaginary answers - process actual data
                    5. Ensure final output is valid JSON using json.dumps()
                    6. Make the code complete and executable

                    COMMON FIXES NEEDED:
                    - Replace placeholder URLs with actual ones from data_summary
                    - Fix file path references to match available files
                    - Add missing imports
                    - Fix syntax errors
                    - Ensure proper JSON output format

                    Return ONLY the corrected Python code (no markdown, no explanations):"""
            )
            fix_prompt += "\nIMPORTANT: If you cannot fix the code without changing the logic, output the original code unchanged."
            
            horizon_fix = await ping_horizon(fix_prompt, "You are a helpful Python code fixer.")
            if "candidates" in horizon_fix:
                fixed_code = horizon_fix["candidates"][0]["content"]["parts"][0]["text"]
            elif "choices" in horizon_fix:
                fixed_code = horizon_fix["choices"][0]["message"]["content"]
            else:
                raise ValueError(f"Unexpected Horizon fix response format: {horizon_fix}")
            
            # Clean the fixed code
            lines = fixed_code.split('\n')
            clean_lines = []
            in_code_block = False

            for line in lines:
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or (not line.strip().startswith('```') and '```' not in line):
                    clean_lines.append(line)

            cleaned_fixed_code = '\n'.join(clean_lines).strip()
            
            with open("chatgpt_code.py", "w", encoding="utf-8") as code_file:
                code_file.write(cleaned_fixed_code)
            # Remove any 'quality=' parameter from plt.savefig or fig.savefig calls
            try:
                with open("chatgpt_code.py", "r", encoding="utf-8") as _f:
                    _code = _f.read()
                _code = re.sub(r'(savefig\s*\([^)]*?),\s*quality\s*=\s*[^,)]+', r'\1', _code)
                with open("chatgpt_code.py", "w", encoding="utf-8") as _f:
                    _f.write(_code)
            except Exception as _e:
                print(f"Warning: failed to clean 'quality=' from savefig (fix): {_e}")

            # Test the fixed code
            result = subprocess.run(
                ["python", "chatgpt_code.py"],
                capture_output=True,
                text=True,
                timeout=120
            )
            # Check for missing module error and try to install
            missing_module = None
            if result.returncode != 0:
                match = re.search(r"No module named '([^']+)'", result.stderr)
                if match:
                    missing_module = match.group(1)
                    print(f"⚠️ Detected missing module during fix code test: {missing_module}. Attempting to install...")
                    try:
                        subprocess.run(
                            ["pip", "install", missing_module],
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        # Re-run the script after installing the module
                        result = subprocess.run(
                            ["python", "chatgpt_code.py"],
                            capture_output=True,
                            text=True,
                            timeout=120
                        )
                    except Exception as e:
                        print(f"❌ Failed to install missing module {missing_module}: {e}")

            if result.returncode == 0:
                stdout = result.stdout.strip()
                json_output = extract_json_from_output(stdout)
                
                if is_valid_json_output(json_output):
                    try:
                        output_data = json.loads(json_output)
                        print(f"✅ Code fixed and executed successfully on fix attempt {fix_attempt}")
                        return output_data
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error on fix attempt {fix_attempt}: {str(e)[:100]}")
                else:
                    print(f"Output still doesn't look like JSON on fix attempt {fix_attempt}: {json_output[:100]}")
            else:
                print(f"Execution still failing on fix attempt {fix_attempt}: {result.stderr}")

        except subprocess.TimeoutExpired:
            print(f"Code execution timed out on fix attempt {fix_attempt}")
        except Exception as e:
            print(f"Unexpected error on fix attempt {fix_attempt}: {e}")

    # If all attempts fail
    return {"error": "Code execution failed after all attempts", "time": time.time() - time_start}