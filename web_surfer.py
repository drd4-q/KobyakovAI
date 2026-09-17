# -*- coding: utf-8 -*-
"""
============================================================
  🌐 KobyakovAI Controlled Web Surfer & Search Engine
  Provides transparent, controlled internet browsing and
  knowledge retrieval for the 144-node MoE AI.
============================================================
"""

import re
import html
import time
import urllib.parse
import requests

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

        # 2. Фолбэк / дополнение из Wikipedia API
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

    def fetch_url(self, url, max_chars=3500):
        """Безопасно загружает и извлекает читаемый текст веб-страницы."""
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        try:
            resp = requests.get(url, headers=self.headers, timeout=8)
            resp.encoding = resp.apparent_encoding or 'utf-8'
            if resp.status_code != 200:
                return f"[Ошибка загрузки страницы: HTTP {resp.status_code}]"

            raw_html = resp.text

            title_m = re.search(r'<title[^>]*>(.*?)</title>', raw_html, re.IGNORECASE | re.DOTALL)
            title = html.unescape(title_m.group(1)).strip() if title_m else url

            cleaned = re.sub(r'<script[^>]*>.*?</script>', ' ', raw_html, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r'<style[^>]*>.*?</style>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r'<noscript[^>]*>.*?</noscript>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r'<nav[^>]*>.*?</nav>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r'<footer[^>]*>.*?</footer>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r'<header[^>]*>.*?</header>', ' ', cleaned, flags=re.IGNORECASE | re.DOTALL)

            text = re.sub(r'<[^>]+>', ' ', cleaned)
            text = html.unescape(text)

            lines = [re.sub(r'\s+', ' ', line).strip() for line in text.splitlines()]
            meaningful_lines = [l for l in lines if len(l) > 30]

            final_text = '\n'.join(meaningful_lines)
            if len(final_text) > max_chars:
                final_text = final_text[:max_chars] + '\n...\n[Текст страницы сокращен]'

            self.last_page_title = title
            self.last_page_url = url
            self.last_page_content = final_text
            self.history.append({
                'type': 'visit',
                'url': url,
                'title': title,
                'time': time.strftime('%H:%M:%S')
            })

            return final_text or "[На странице не обнаружено читаемого текста]"

        except Exception as e:
            return f"[Ошибка при переходе по ссылке: {e}]"

    def format_search_context(self, query, results):
        """Форматирует факты из поиска для контекста MoE нейросети."""
        if not results:
            return ""

        lines = [f"[🌐 Данные веб-поиска по запросу: «{query}»]"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']} ({r.get('domain', 'web')}):")
            lines.append(f"   {r['snippet']}")
        return '\n'.join(lines) + '\n\n'
