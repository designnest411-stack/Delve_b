"""
Comprehensive Service Health & Integration Test
────────────────────────────────────────────────
Validates end-to-end connectivity across:
  1. Tavily Academic Web Search API
  2. Google Gemini Dual-Model LLM Engine (with rate-pacing)
  3. Supabase PostgreSQL Tables (RLS enabled) and Storage Bucket
"""

import asyncio
import os
import sys
from pathlib import Path
import httpx

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.core.llm_client import llm_client
from app.services.retrieval import fetch_web_tavily
from app.core.supabase import supabase_repository

async def test_tavily():
    print("=" * 60)
    print("1. TESTING TAVILY SEARCH API")
    print("=" * 60)
    key = settings.tavily_api_key
    print(f"Key configured: {key[:8]}...{key[-4:] if len(key)>12 else ''} (length: {len(key)})")
    
    try:
        from tavily import AsyncTavilyClient
        client = AsyncTavilyClient(api_key=key)
        res = await client.search(query="latest advances in artificial intelligence 2026", max_results=2)
        results = res.get("results", [])
        print(f"[OK] Direct Tavily search returned {len(results)} results:")
        for idx, r in enumerate(results, 1):
            print(f"   {idx}. {r.get('title')} ({r.get('url')})")
            
        retrieval_res = await fetch_web_tavily("quantum computing algorithms", key, max_results=2)
        print(f"[OK] fetch_web_tavily returned {len(retrieval_res)} standardized paper/web dicts.")
        return True, "Tavily search is operational."
    except Exception as e:
        print(f"[FAIL] Tavily test failed: {e}")
        return False, str(e)

async def test_llm():
    print("\n" + "=" * 60)
    print("2. TESTING GEMINI LLM API")
    print("=" * 60)
    key = settings.gemini_api_key
    model = settings.gemini_primary_model
    print(f"Key configured: {key[:8]}...{key[-4:] if len(key)>12 else ''} (length: {len(key)})")
    print(f"Primary model: {model}")
    print(f"Fallback cascade: {settings.gemini_fallback_models}")

    try:
        valid = await llm_client.validate_api_key()
        print(f"llm_client.validate_api_key() result: {valid}")
        prompt = "Provide a 1-sentence confirmation that the Gemini engine is online and ready for research."
        resp = await llm_client.generate_content(prompt)
        print(f"[OK] Gemini response received:\n   \"{resp.strip()}\"")
        return True, resp.strip()
    except Exception as e:
        print(f"[FAIL] Gemini LLM test failed: {e}")
        return False, str(e)

async def test_supabase():
    print("\n" + "=" * 60)
    print("3. TESTING SUPABASE DATABASE & STORAGE")
    print("=" * 60)
    url = settings.supabase_url
    bucket = settings.supabase_storage_bucket
    print(f"Supabase URL: {url}")
    print(f"Storage Bucket: {bucket}")

    results = {}
    
    tables = [
        "research_sessions",
        "research_jobs",
        "research_documents",
        "document_chunks",
        "user_paper_quotas",
        "llm_usage"
    ]

    for table in tables:
        try:
            rows = await supabase_repository._rest("GET", table, params={"select": "*", "limit": "1"})
            results[f"table:{table}"] = f"OK (accessible, {len(rows)} sample row(s) inspected)"
            print(f"[OK] Table '{table}' accessible.")
        except Exception as e:
            results[f"table:{table}"] = f"FAIL: {e}"
            print(f"[FAIL] Table '{table}': {e}")

    try:
        headers = supabase_repository._headers()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{settings.supabase_storage_url}/bucket/{bucket}", headers=headers)
        if resp.status_code == 200:
            results["storage_bucket"] = f"OK (bucket '{bucket}' exists and is reachable)"
            print(f"[OK] Storage bucket '{bucket}' exists and is reachable.")
        else:
            results["storage_bucket"] = f"Warning: HTTP {resp.status_code} - {resp.text}"
            print(f"[!] Storage bucket response: {resp.status_code} {resp.text}")
    except Exception as e:
        results["storage_bucket"] = f"FAIL: {e}"
        print(f"[FAIL] Storage bucket: {e}")

    return results

async def main():
    tavily_ok, tavily_msg = await test_tavily()
    llm_ok, llm_msg = await test_llm()
    db_results = await test_supabase()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Tavily Search: {'PASS' if tavily_ok else 'FAIL'}")
    print(f"Gemini LLM:    {'PASS' if llm_ok else 'FAIL'}")
    all_tables_ok = all(v.startswith("OK") for k, v in db_results.items() if k.startswith("table:"))
    print(f"Supabase DB:   {'PASS' if all_tables_ok else 'FAIL'}")

if __name__ == "__main__":
    asyncio.run(main())
