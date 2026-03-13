# start_server.ps1
# Starts the ChromaDB-backed News Analyst agent server.
# Run this in Terminal 1 BEFORE running arksim in Terminal 2.
#
# Usage:
#   cd D:\SpringBigData\arksim\examples\news-analyst
#   .\start_server.ps1

Write-Host ""
Write-Host "============================================================"
Write-Host " News Analyst Agent Server - ChromaDB Backend"
Write-Host "============================================================"
Write-Host ""

# Check OPENAI_API_KEY
if (-not $env:OPENAI_API_KEY) {
    Write-Host "ERROR: OPENAI_API_KEY is not set."
    Write-Host "  Run: " -NoNewline
    Write-Host '$env:OPENAI_API_KEY="sk-..."'
    exit 1
}

# Check ChromaDB index exists
$dbPath = "agent_server\VectorDB\chroma.sqlite3"
if (-not (Test-Path $dbPath)) {
    Write-Host "ERROR: ChromaDB index not found at $dbPath"
    Write-Host "  Run first: python build_index.py --csv agent_server/data/articles.csv"
    exit 1
}

# Set server API key (must match config_chromadb.yaml AGENT_API_KEY)
if (-not $env:AGENT_API_KEY) {
    $env:AGENT_API_KEY = "chromadb-secret"
    Write-Host "AGENT_API_KEY not set - using default: chromadb-secret"
}

Write-Host "Starting server on http://localhost:8888 ..."
Write-Host "  OPENAI_API_KEY : $($env:OPENAI_API_KEY.Substring(0,12))..."
Write-Host "  AGENT_API_KEY  : $env:AGENT_API_KEY"
Write-Host "  ChromaDB index : $dbPath"
Write-Host ""
Write-Host "Keep this terminal open. In Terminal 2 run:"
Write-Host "  arksim simulate-evaluate config_chromadb.yaml"
Write-Host ""

python -m agent_server.chat_completions.server
