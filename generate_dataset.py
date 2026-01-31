import os
import time
import re
import requests
import argparse
import json
import logging
import base64
import shutil
from gradio_client import Client
from gradio_client.exceptions import AppError
import filelock

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TTS dataset for Ukrainian voice training"
    )
    parser.add_argument("--start", type=int, default=0,
                        help="Start sample index")
    parser.add_argument("--end", type=int, default=None,
                        help="End sample index (exclusive)")
    parser.add_argument("--delay", type=int, default=1,
                        help="Delay between requests in seconds")
    parser.add_argument("--local", action="store_true",
                        help="Use local Docker container instead of HuggingFace Space")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:7860",
                        help="Local API URL (default: http://127.0.0.1:7860)")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration of existing samples")
    return parser.parse_args()

def generate_dataset():
    args = parse_args()
    start_idx = args.start
    end_idx = args.end
    delay = args.delay
    use_local = args.local
    api_url = args.api_url.rstrip("/")

    # Configuration
    SPACE_ID = "patriotyk/styletts2-ukrainian"
    OUTPUT_DIR = "dataset_marina"
    # Derive samples from sentences list
    # We want ~1000 for training, so we repeat the 100 unique ones 10 times
    # This gives the model more 'epochs' per step if data is small
    pass
    
    # TTS Parameters
    MODEL_NAME = "multi"
    VOICE_NAME = "Марина Панас"
    SPEED = 1.3
    
    # Diverse Sentences to synthesize (Ukrainian)
    # 100+ unique sentences covering various phonemes
    sentences = [
        "Привіт, це голос Марини Панас, згенерований штучним інтелектом.",
        "Україна — це велика країна з багатою культурою та історією.",
        "Сьогодні ми навчимо комп'ютер розмовляти нашою мовою.",
        "Сонце світить яскраво, і пташки співають у садку.",
        "Технології майбутнього приносять користь кожній людині.",
        "Київ — серце України, де поєднується давнина та сучасність.",
        "Діти граються на подвір'ї та радіють літньому теплу.",
        "Мирного неба над головою та спокою у кожній домівці.",
        "Знання — це сила, а книга — джерело мудрості.",
        "Дякуємо за допомогу та підтримку у цей складний час.",
        "Смачна кава зранку дарує бадьорість на весь робочий день.",
        "Дніпро несе свої води через усю неосяжну країну.",
        "Смарагдові ліси Карпат чарують своєю спокійною величчю.",
        "Мова — це не просто засіб спілкування, а душа народу.",
        "Кожен крок до мети робить нас сильнішими та впевненішими.",
        "Весна приходить непомітно, розцвітаючи першими квітами.",
        "Осінь фарбує листя у золоті та червоні візерунки.",
        "Зима дарує нам снігові розваги та святковий настрій.",
        "Літо — це час для подорожей та нових відкриттів.",
        "Ми прагнемо досконалості у всьому, що ми робимо разом.",
        "Важливо пам'ятати свою історію та шанувати предків.",
        "Музика об'єднує серця та дарує надію на краще майбутнє.",
        "Спорт допомагає тримати тіло в тонусі та загартовує дух.",
        "Здоров'я — найцінніший скарб, який потрібно берегти змолоду.",
        "Наука відкриває нові обрії та робить неможливе реальним.",
        "Мистецтво надихає на творчість та вчить бачити красу навколо.",
        "Праця — основа добробуту та розвитку будь-якої держави.",
        "Сім'я — це опора та підтримка у будь-яких життєвих обставинах.",
        "Дружба дарує радість спілкування та вірність у випробуваннях.",
        "Кохання робить світ кращим та наповнює життя глибоким змістом.",
        "Чесність та порядність — фундамент міцних стосунків між людьми.",
        "Повага до оточуючих створює атмосферу гармонії та взаєморозуміння.",
        "Відповідальність перед майбутніми поколіннями — обов'язок кожного.",
        "Ми дбаємо про довкілля та прагнемо зберегти природу для дітей.",
        "Інновації змінюють наше життя на краще кожного нового дня.",
        "Освіта — це ключ до успішної кар'єри та самореалізації особистості.",
        "Читати книги — значить відкривати для себе тисячі нових життів.",
        "Подорожі розширюють світогляд та вчать цінувати кожну мить.",
        "Ми впевнено дивимося в майбутнє та віримо у свої сили завжди.",
        "Разом ми зможемо подолати будь-які труднощі та виклики часу.",
        "Успіх приходить до тих, хто не боїться падати та підніматися.",
        "Мрії збуваються, якщо наполегливо йти до своєї мети щодня.",
        "Вдячність — це ознака шляхетного серця та чистої душі.",
        "Сміливість допомагає перемагати страх та відкривати нові шляхи.",
        "Доброта — це мова, яку розуміють навіть ті, хто не чує.",
        "Милосердя робить нас людьми та наповнює світ світлом любові.",
        "Справедливість — це основа правової держави та вільного суспільства.",
        "Свобода — найвище благо, за яке варто боротися до кінця.",
        "Ми пишаємося своєю незалежністю та захищаємо свої кордони.",
        "Слава Україні! Героям слава! Разом до перемоги у майбутньому!",
        "Ніч приходить тихо, запалюючи перші зорі на вечірньому небі.",
        "Місяць кидає своє срібне сяйво на спляче сонне місто.",
        "Ранкова роса виблискує на траві, немов казкові маленькі діаманти.",
        "Мелодія скрипки лунає у порожній залі, зачіпаючи струни душі.",
        "Вітер шепоче щось таємниче у кронах старих могутніх дубів.",
        "Дощ стукає у вікно, навіваючи спокій та легкий смуток самотності.",
        "Аромат квітів наповнює повітря, створюючи атмосферу весняної казки.",
        "Смарагдовий берег моря манить своєю свіжістю та безкрайньою даллю.",
        "Гірські річки течуть швидко, долаючи всі перешкоди на своєму шляху.",
        "Високе небо кличе мрійників до нових космічних неосяжних вершин.",
        "Кожен вечір — це можливість подумати про пережите за весь день сьогодні.",
        "Ранкове тренування дає заряд енергії на багато плідних годин вперед.",
        "Свіжі овочі та фрукти — запорука правильного здорового харчування.",
        "Затишний будинок — місце, куди завжди хочеться повернутися з дороги.",
        "Гарний фільм допомагає відволіктися від буденності та зануритися в екшн.",
        "Творчий підхід до справи дозволяє знаходити нестандартні проривні рішення.",
        "Увага до деталей — важлива складова професіоналізму у будь-якій сфері.",
        "Командна робота забезпечує високий результат та швидке виконання завдань.",
        "Креативне мислення допомагає бачити можливості там, де інші бачать проблеми.",
        "Систематичне навчання веде до майстерності та високих особистих досягнень.",
        "Гнучкість у прийнятті рішень дозволяє адаптуватися до будь-яких швидких змін.",
        "Аналітичні здібності допомагають приймати зважені та обґрунтовані рішення.",
        "Стратегічне планування — ключ до сталого розвитку та процвітання бізнесу.",
        "Лідерські якості надихають інших на великі звершення та нові перемоги.",
        "Емоційний інтелект допомагає краще розуміти себе та оточуючих нас колег.",
        "Ефективна комунікація будує міцні зв'язки та сприяє успіху у справах.",
        "Критичне мислення захищає від маніпуляцій та вчить розрізняти правду і фейк.",
        "Постійний розвиток — єдиний шлях до вершини у сучасному динамічному світі.",
        "Ми створюємо продукти, якими люди будуть із задоволенням користуватися щодня.",
        "Якість обслуговування — наш головний пріоритет та запорука лояльності клієнтів.",
        "Ми цінуємо кожного співробітника та створюємо умови для зростання колективу.",
        "Корпоративна культура об'єднує нас та допомагає разом йти вперед.",
        "Соціальна відповідальність бізнесу сприяє покращенню життя всієї країни.",
        "Екологічні ініціативи допомагають зберігати планету чистою для нащадків.",
        "Доброчинність — це внесок у краще майбутнє для тих, хто потребує допомоги.",
        "Волонтерський рух демонструє силу згуртованості та небайдужості українців.",
        "Підтримка армії — це наш спільний обов'язок перед захисниками Вітчизни.",
        "Разом ми — незламна сила, здатна на будь-які героїчні та великі вчинки.",
        "Український дух неможливо зламати, бо за нами правда та божа опіка.",
        "Перемога буде за нами, бо ми на своїй землі та захищаємо свій рід.",
        "Майбутнє України — серед вільних та демократичних націй всього світу.",
        "Ми пишаємося своїм минулим та впевнено будуємо своє щасливе майбутнє.",
        "Віра у краще допомагає виживати у самі найтемніші часи нашого життя.",
        "Надія вмирає останньою, але ми зробимо все, щоб вона жила вічно.",
        "Любов переможе все, бо вона є найсильнішою магією у цілому всесвіті.",
        "Будьте щасливі, любіть одне одного та цінуйте кожну прожиту хвилину.",
        "Світ чекає на ваші ідеї, таланти та великі звершення вже сьогодні увечері.",
        "Сміливо йдіть вперед, не озираючись на минулі помилки та невдачі.",
        "Ви — автори своєї долі, тому пишіть свою історію сміливо та яскраво.",
        "Ви — автори своєї долі, тому пишіть свою історію сміливо та яскраво.",
        "Нехай доля вам посміхається, а успіх завжди супроводжує ваші починання.",
    ] * 10 # ~1000 samples (100 unique repeated 10 times)
    
    # Slice sentences based on start/end args for distributed generation
    if end_idx is not None:
        sentences = sentences[:end_idx]
    if start_idx > 0:
        sentences = sentences[start_idx:]
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    if use_local:
        # Use local Docker container - Gradio API format
        gradio_api_url = f"{api_url}/gradio_api"
        synthesize_url = f"{gradio_api_url}/call/synthesize"
        logger.info(f"Using local Docker at {api_url}")
        logger.info(f"Gradio API URL: {gradio_api_url}")
        print(f"Using local Docker at {api_url}")
        print(f"No quota limits - local GPU is yours!\n")
        client = None
    else:
        # Use HuggingFace Space
        print(f"Connecting to {SPACE_ID}...")
        try:
            client = Client(SPACE_ID)
        except Exception as e:
            print(f"Failed to connect: {e}")
            return
        synthesize_url = None
    
    # Calculate actual range for display
    actual_start = start_idx
    actual_end = end_idx if end_idx is not None else len(sentences) + start_idx
    print(f"Generating samples {actual_start} to {actual_end-1} ({len(sentences)} total)...")
    print(f"Delay between requests: {delay}s\n")
    
    filelist_path = os.path.join(OUTPUT_DIR, "filelist.txt")
    lock = filelock.FileLock(os.path.join(OUTPUT_DIR, "filelist.lock"))
    
    # Read existing entries to avoid duplicates
    existing_files = set()
    existing_entries = set()
    if os.path.exists(filelist_path):
        with open(filelist_path, "r", encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    existing_entries.add(line.strip())
                    path_part = line.split("|")[0]
                    existing_files.add(os.path.basename(path_part))
    
    for i, text in enumerate(sentences):
        global_idx = actual_start + i
        filename = f"sample_{global_idx:04d}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        abs_path = os.path.abspath(filepath)
        entry = f"{abs_path}|{text}"
        
        # Skip if already processed (unless --force)
        file_exists = os.path.exists(filepath)
        if not args.force:
            if filename in existing_files:
                print(f"[{i+1}/{len(sentences)}] Skipping {global_idx}, already in filelist.")
                continue
            elif file_exists and entry in existing_entries:
                print(f"[{i+1}/{len(sentences)}] Skipping {global_idx}, file exists and entry matches.")
                continue
            elif file_exists:
                print(f"[{i+1}/{len(sentences)}] File exists but not in filelist. Adding entry...")
                with lock:
                    with open(filelist_path, "a", encoding="utf-8") as f_list:
                        f_list.write(f"{abs_path}|{text}\n")
                existing_entries.add(entry)
                existing_files.add(filename)
                continue
        
        print(f"[{i+1}/{len(sentences)}] Generating sample_{global_idx}: {text[:30]}...")
        
        success = False
        audio_path = None
        
        try:
            if use_local:
                # Use local Docker API - Gradio API format with SSE streaming
                # API signature: synthesize(model_name, text, speed, voice_name)
                payload = {
                    "data": [
                        MODEL_NAME,
                        text,
                        SPEED,
                        VOICE_NAME,
                    ]
                }
                
                logger.info(f"Calling {synthesize_url} with payload: {payload}")
                
                # Step 1: Submit synthesis request
                response = requests.post(
                    synthesize_url,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code != 200:
                    logger.error(f"API call failed with {response.status_code}: {response.text}")
                    raise Exception(f"API call failed: {response.status_code}")
                
                result = response.json()
                event_id = result.get("event_id")
                if not event_id:
                    raise Exception(f"No event_id in response: {result}")
                
                logger.info(f"Event ID received: {event_id}")
                
                # Step 2: Stream the result
                stream_url = f"{synthesize_url}/{event_id}"
                audio_data = None
                
                with requests.get(stream_url, stream=True, timeout=60) as stream_resp:
                    for line in stream_resp.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith("event: "):
                                current_event = line_str[7:].strip()
                                logger.debug(f"SSE event: {current_event}")
                            elif line_str.startswith("data: "):
                                data_str = line_str[6:].strip()
                                try:
                                    data = json.loads(data_str)
                                    
                                    if current_event in ("output", "complete"):
                                        logger.info(f"Received audio data (event: {current_event})")
                                        # Gradio returns data as a list
                                        results = data if isinstance(data, list) else data.get("data", [])
                                        if results and len(results) > 0 and results[0] is not None:
                                            audio_data = results[0]
                                            break
                                except json.JSONDecodeError:
                                    continue
                
                if audio_data is None:
                    raise Exception("No audio data received from stream")
                
                # Step 3: Download audio file
                # audio_data can be a dict with url/path or base64 string
                if isinstance(audio_data, dict):
                    file_url = audio_data.get("url") or audio_data.get("path")
                    if file_url:
                        if not file_url.startswith("http"):
                            file_url = f"{api_url}/file={file_url}"
                        logger.info(f"Downloading audio from {file_url}")
                        audio_response = requests.get(file_url, timeout=60)
                        audio_response.raise_for_status()
                        audio_bytes = audio_response.content
                    else:
                        raise Exception("No url/path in audio data")
                elif isinstance(audio_data, str):
                    # Base64 encoded
                    if audio_data.startswith("data:audio"):
                        _, encoded = audio_data.split(",", 1)
                        audio_bytes = base64.b64decode(encoded)
                    else:
                        audio_bytes = base64.b64decode(audio_data)
                else:
                    raise Exception(f"Unknown audio data format: {type(audio_data)}")
                
                # Save audio file
                audio_path = filepath
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)
                
                logger.info(f"Audio saved to {audio_path}")
                
            else:
                # Use HuggingFace Space via gradio_client
                # Implementation with Retries for Quota/Errors
                max_retries = 10
                retry_delay = 10 # Start with 10s
                
                for attempt in range(max_retries):
                    try:
                        result = client.predict(
                            MODEL_NAME,
                            text,
                            1.0,
                            VOICE_NAME,
                            api_name="/synthesize"
                        )
                        
                        if isinstance(result, (list, tuple)):
                            audio_path = result[0] if len(result) > 0 else None
                        elif isinstance(result, dict):
                            audio_path = result.get('audio') or result.get('file') or result.get('path')
                        else:
                            audio_path = result
                        
                        if audio_path and os.path.exists(audio_path):
                            shutil.copy(audio_path, filepath)
                            audio_path = filepath
                            break # Success!
                        else:
                            raise Exception("Invalid audio path returned")
                            
                    except AppError as e:
                        if "quota" in str(e).lower():
                            print(f"  ⚠ Quota Exceeded. Attempt {attempt+1}/{max_retries}. Sleeping {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay *= 2 # Exponential backoff
                        else:
                            print(f"  ✗ App Error: {e}")
                            break
                    except Exception as e:
                        print(f"  ✗ Unexpected Error: {e}")
                        break
            
            if audio_path and os.path.exists(audio_path):
                print(f"  ✓ Saved to {filename}")
                
                if entry not in existing_entries:
                    with lock:
                        with open(filelist_path, "a", encoding="utf-8") as f_list:
                            f_list.write(f"{abs_path}|{text}\n")
                    existing_entries.add(entry)
                    existing_files.add(filename)
                success = True
                
        except Exception as e:
            print(f"  ✗ Error generating sample {global_idx}: {e}")
            logger.error(f"Error generating sample {global_idx}: {e}", exc_info=True)
        
        if success and i < len(sentences) - 1:
            time.sleep(delay)
    
    # Count successful generations
    successful = sum(1 for i in range(len(sentences)) 
                     if os.path.exists(os.path.join(OUTPUT_DIR, f"sample_{actual_start + i:04d}.wav")))
    
    print(f"\n{'='*60}")
    print(f"Done! Dataset saved to {OUTPUT_DIR}")
    print(f"Successfully generated: {successful}/{len(sentences)} samples (global indices {actual_start} to {actual_start + len(sentences) - 1})")
    print(f"Filelist: {filelist_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    generate_dataset()
