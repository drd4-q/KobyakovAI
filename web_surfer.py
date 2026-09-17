# -*- coding: utf-8 -*-
"""
============================================================
  🌐 KobyakovAI Controlled Web Surfer & Semantic Reasoning Engine
  Provides transparent internet browsing, 100% full-page reading,
  and Semantic Meaning-Based Selection (отбор по смыслу).
============================================================
"""

import re
import html
import time
import urllib.parse
import requests

def extract_full_page_text(raw_html):
    """
    Извлекает полный читаемый текст страницы БЕЗ ОГРАНИЧЕНИЙ:
    - Сохраняет иерархию заголовков (h1-h6 -> ### Заголовок)
    - Сохраняет абзацы, таблицы, списки (• item)
    - Очищает от скриптов, стилей, SVG, навигации и подвалов
    - 100% объем страницы без обрезания
    """
    title_m = re.search(r'<title[^>]*>(.*?)</title>', raw_html, re.IGNORECASE | re.DOTALL)
    title = html.unescape(title_m.group(1)).strip() if title_m else ""

    # Удаляем нерелевантные теги
    cleaned = re.sub(r'<script[^>]*>.*?</script>', ' ', raw_html, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<style[^>]*>.*?</style>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<noscript[^>]*>.*?</noscript>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<nav[^>]*>.*?</nav>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<footer[^>]*>.*?</footer>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<header[^>]*>.*?</header>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<svg[^>]*>.*?</svg>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<iframe[^>]*>.*?</iframe>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Сохраняем структуру
    cleaned = re.sub(r'<h[1-3][^>]*>', '\n\n### ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<h[4-6][^>]*>', '\n\n#### ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<(p|div|tr|blockquote)[^>]*>', '\n\n', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<br\s*/?>', '\n', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'<li[^>]*>', '\n• ', cleaned, flags=re.IGNORECASE)

    # Удаляем остальные теги и декодируем сущности
    text = re.sub(r'<[^>]+>', ' ', cleaned)
    text = html.unescape(text)

    # Нормализуем строки
    lines = []
    for line in text.splitlines():
        line = re.sub(r'[ \t]+', ' ', line).strip()
        if line and line != '•' and line != '• ':
            lines.append(line)

    full_text = '\n'.join(lines)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    return title, full_text


class SemanticMeaningSelector:
    """
    Движок семантического анализа и смыслового отбора:
    Анализирует интент вопроса (определение, факт, число, правила, причина),
    разбивает гигантский текст страницы на смысловые блоки и выбирает
    именно те абзацы, которые отвечают на вопрос пользователя.
    """
    STOP_WORDS = {
        'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она',
        'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее',
        'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда',
        'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до',
        'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей',
        'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем',
        'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет',
        'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь',
        'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем',
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with'
    }

    @classmethod
    def extract_keywords(cls, query):
        words = re.findall(r'[A-Za-zА-Яа-я0-9_-]+', query.lower())
        return [w for w in words if w not in cls.STOP_WORDS and len(w) > 2]

    @classmethod
    def detect_intent(cls, query):
        q_low = query.lower()
        intent = {
            'wants_number': bool(re.search(r'(число|числа|номер|год|году|года|сколько|когда|дата|дате|статистик|век|number|year|date|how many|when|count)', q_low)),
            'wants_definition': bool(re.search(r'(что такое|кто такой|что значит|определени|суть|термин|поняти|значени|what is|who is|meaning|definition)', q_low)),
            'wants_rules': bool(re.search(r'(как|правил|механик|устройств|принцип|процесс|работает|игра|роль|how|rules|mechanics|system)', q_low)),
            'wants_cause': bool(re.search(r'(почему|зачем|причин|откуда|истори|возникн|why|cause|origin|history)', q_low))
        }
        return intent

    @classmethod
    def select_relevant_blocks(cls, query, full_text, top_n=4):
        if not full_text or not query:
            return []

        keywords = cls.extract_keywords(query)
        intent = cls.detect_intent(query)

        # Разбиваем текст на параграфы (разделенные двойным переводом строки)
        raw_paragraphs = full_text.split('\n\n')
        scored_paragraphs = []

        current_heading = "Общий раздел"

        for p in raw_paragraphs:
            p = p.strip()
            if not p:
                continue

            # Отслеживаем заголовки
            if p.startswith('### ') or p.startswith('#### '):
                current_heading = p.lstrip('#').strip()
                continue

            # Пропускаем очевидный мусор
            p_low = p.lower()
            if any(noise in p_low for noise in ['политика конфиденциальности', 'cookie', 'соглашение с пользователем', 'навигационное меню', 'условия использования']):
                continue
            if len(p) < 30:
                continue

            score = 0.0

            # 1. Совпадение по ключевым словам
            kw_hits = 0
            for kw in keywords:
                if kw in p_low:
                    kw_hits += 1
                    score += 2.0
            
            # Бонус за покрытие уникальных ключевых слов
            if keywords:
                coverage = kw_hits / len(keywords)
                score += coverage * 4.0

            # 2. Интент: число / дата / год
            if intent['wants_number']:
                digits_found = len(re.findall(r'\b\d+\b', p))
                if digits_found > 0:
                    score += min(digits_found * 1.5, 6.0)
                if any(w in p_low for w in ['год', 'году', 'век', 'число', 'дата']):
                    score += 3.0

            # 3. Интент: определение
            if intent['wants_definition']:
                if any(m in p for m in ['— это', '— разновидность', 'является', 'определяется', 'представляет собой', 'называют', 'от англ.', 'is defined as', 'refers to']):
                    score += 5.0

            # 4. Интент: правила / механика
            if intent['wants_rules']:
                if any(m in p_low for m in ['правил', 'отыгрыш', 'персонаж', 'действие', 'мастер', 'роль', 'участник']):
                    score += 4.0

            # 5. Интент: происхождение / причина
            if intent['wants_cause']:
                if any(m in p_low for m in ['происходит от', 'возник', 'создан', 'причина', 'впервые']):
                    score += 4.0

            # Бонус за релевантность текущего заголовка
            h_low = current_heading.lower()
            for kw in keywords:
                if kw in h_low:
                    score += 2.5

            if score > 0.5:
                scored_paragraphs.append({
                    'heading': current_heading,
                    'text': p,
                    'score': score
                })

        # Сортируем по убыванию смысловой релевантности
        scored_paragraphs.sort(key=lambda x: x['score'], reverse=True)
        return scored_paragraphs[:top_n]


class WebSurfer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        }
        self.history = []
        self.last_query = ""
        self.last_results = []
        self.last_page_content = ""
        self.last_page_title = ""
        self.last_page_url = ""
        self.last_page_chars = 0
        self.last_page_lines = 0

    @staticmethod
    def extract_urls(text):
        """Находит прямые ссылки в тексте."""
        return [w.rstrip('.,;!?:') for w in re.findall(r'https?://[^\s<>"]+', text)]

    def search(self, query, max_results=4):
        """Выполняет контролируемый поиск через DuckDuckGo и Wikipedia."""
        query = query.strip()
        if not query:
            return []

        self.last_query = query
        results = []

        # 1. Поиск через DuckDuckGo HTML
        try:
            url = 'https://html.duckduckgo.com/html/'
            resp = requests.post(url, data={'q': query}, headers=self.headers, timeout=6)
            if resp.status_code == 200:
                raw_results = re.findall(r'<a class="result__snippet"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', resp.text, re.DOTALL)
                titles = re.findall(r'<h2 class="result__title">\s*<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', resp.text, re.DOTALL)
                
                title_map = {}
                for t_url, t_text in titles:
                    clean_t = html.unescape(re.sub(r'<[^>]+>', '', t_text)).strip()
                    title_map[t_url.strip()] = clean_t

                for raw_link, raw_snippet in raw_results[:max_results]:
                    snippet = html.unescape(re.sub(r'<[^>]+>', '', raw_snippet)).strip()
                    link = raw_link.strip()
                    if "uddg=" in link:
                        link = urllib.parse.unquote(link.split("uddg=")[-1].split("&")[0])
                    
                    domain = "web"
                    try:
                        domain = urllib.parse.urlparse(link).netloc.replace('www.', '')
                    except Exception:
                        pass

                    title = title_map.get(raw_link.strip(), f"Источник ({domain})")
                    if snippet and len(snippet) > 10:
                        results.append({
                            'title': title,
                            'url': link,
                            'domain': domain,
                            'snippet': snippet
                        })
        except Exception as e:
            print(f"[WebSurfer] DDG search error: {e}")

        # 2. Фолбэк / дополнение из Wikipedia API (русская)
        if len(results) < 2:
            try:
                wiki_url = f"https://ru.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(query)}&limit=3&namespace=0&format=json"
                w_resp = requests.get(wiki_url, headers=self.headers, timeout=5)
                if w_resp.status_code == 200:
                    data = w_resp.json()
                    if len(data) >= 4 and data[1]:
                        for i in range(len(data[1])):
                            w_title = data[1][i]
                            w_desc = data[2][i] if i < len(data[2]) else ""
                            w_link = data[3][i] if i < len(data[3]) else ""
                            if w_desc:
                                results.append({
                                    'title': f"Википедия: {w_title}",
                                    'url': w_link,
                                    'domain': 'ru.wikipedia.org',
                                    'snippet': w_desc
                                })
            except Exception as e:
                print(f"[WebSurfer] Wikipedia search error: {e}")

        # 3. Английская Википедия для англоязычных запросов
        if len(results) < 2 and any(ord(c) < 128 and c.isalpha() for c in query):
            try:
                wiki_en = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(query)}&limit=3&namespace=0&format=json"
                w_resp = requests.get(wiki_en, headers=self.headers, timeout=5)
                if w_resp.status_code == 200:
                    data = w_resp.json()
                    if len(data) >= 4 and data[1]:
                        for i in range(len(data[1])):
                            w_title = data[1][i]
                            w_desc = data[2][i] if i < len(data[2]) else ""
                            w_link = data[3][i] if i < len(data[3]) else ""
                            if w_desc:
                                results.append({
                                    'title': f"Wikipedia: {w_title}",
                                    'url': w_link,
                                    'domain': 'en.wikipedia.org',
                                    'snippet': w_desc
                                })
            except Exception:
                pass

        self.last_results = results[:max_results]
        self.history.append({
            'type': 'search',
            'query': query,
            'count': len(self.last_results),
            'time': time.strftime('%H:%M:%S')
        })
        if len(self.history) > 20:
            self.history.pop(0)

        return self.last_results

    def fetch_url(self, url, max_chars=None):
        """
        Загружает и возвращает 100% полный текст веб-страницы БЕЗ ОГРАНИЧЕНИЙ:
        - Если max_chars=None, загружается абсолютно весь читаемый текст статьи!
        """
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        try:
            resp = requests.get(url, headers=self.headers, timeout=12)
            resp.encoding = resp.apparent_encoding or 'utf-8'
            if resp.status_code != 200:
                return f"[Ошибка загрузки страницы: HTTP {resp.status_code}]"

            title, full_text = extract_full_page_text(resp.text)
            if not title:
                title = url

            if max_chars is not None and len(full_text) > max_chars:
                full_text = full_text[:max_chars] + f"\n\n[... Сокращено с {len(full_text)} до {max_chars} символов ...]"

            lines_count = len(full_text.splitlines())
            self.last_page_title = title
            self.last_page_url = url
            self.last_page_content = full_text
            self.last_page_chars = len(full_text)
            self.last_page_lines = lines_count

            self.history.append({
                'type': 'visit',
                'url': url,
                'title': title,
                'chars': len(full_text),
                'time': time.strftime('%H:%M:%S')
            })

            return full_text or "[На странице не обнаружено читаемого текста]"

        except Exception as e:
            return f"[Ошибка при переходе по ссылке: {e}]"

    def search_with_deep_crawl(self, query, max_results=3, crawl_top=True):
        """
        Выполняет поиск и глубокий смысловой краулинг:
        - Ищет результаты
        - Загружает полный текст топ-результата (100% объема)
        - Выполняет семантический отбор по смыслу (Semantic Meaning Selection)
        """
        results = self.search(query, max_results=max_results)
        top_page_text = ""
        top_page_title = ""
        top_page_url = ""
        relevant_blocks = []

        if crawl_top and results:
            first_url = results[0].get('url', '')
            if first_url and first_url.startswith(('http://', 'https://')):
                try:
                    print(f"[WebSurfer] Полный краулинг страницы: {first_url}...")
                    top_page_text = self.fetch_url(first_url, max_chars=None)
                    top_page_title = self.last_page_title
                    top_page_url = first_url

                    # Семантический отбор по смыслу из полного объема
                    relevant_blocks = SemanticMeaningSelector.select_relevant_blocks(query, top_page_text, top_n=3)
                except Exception as ce:
                    print(f"[WebSurfer] Ошибка краулинга {first_url}: {ce}")

        return {
            'results': results,
            'top_page_title': top_page_title,
            'top_page_url': top_page_url,
            'top_page_text': top_page_text,
            'relevant_blocks': relevant_blocks
        }
