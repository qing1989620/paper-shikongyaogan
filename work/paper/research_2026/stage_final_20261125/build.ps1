$ErrorActionPreference = 'Stop'
$source = Join-Path $PSScriptRoot 'source'
$pdfDir = Join-Path $PSScriptRoot 'output/pdf'
New-Item -ItemType Directory -Path $pdfDir -Force | Out-Null
Push-Location -LiteralPath $source
try {
    foreach ($lang in @('zh', 'en')) {
        $job = "STF_stage_final_$lang"
        $build = ".build/$lang"
        New-Item -ItemType Directory -Path $build -Force | Out-Null
        & latexmk -xelatex -silent -interaction=nonstopmode -halt-on-error "-outdir=$build" "-jobname=$job" "manuscript_$lang.tex"
        if ($LASTEXITCODE -ne 0) { throw "Compilation failed. See source/$build/$job.log." }
        Copy-Item -LiteralPath "$build/$job.pdf" -Destination (Join-Path $pdfDir "$job.pdf") -Force
    }
} finally {
    Pop-Location
}
