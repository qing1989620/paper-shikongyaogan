$ErrorActionPreference = 'Stop'
$package = $PSScriptRoot
$source = Join-Path $package 'source'
$pdfDir = Join-Path $package 'output/pdf'
New-Item -ItemType Directory -Path $pdfDir -Force | Out-Null
Push-Location -LiteralPath $source
try {
    foreach ($language in @('en', 'zh')) {
        $engine = if ($language -eq 'en') { '-pdf' } else { '-xelatex' }
        $job = "STF_stage_$language"
        $build = ".build/$language"
        New-Item -ItemType Directory -Path $build -Force | Out-Null
        & latexmk $engine -silent -interaction=nonstopmode -halt-on-error "-outdir=$build" "-jobname=$job" "manuscript_$language.tex"
        if ($LASTEXITCODE -ne 0) { throw "TeX compilation failed for $language. See source/.build/$language/$job.log." }
        Copy-Item -LiteralPath "$build/$job.pdf" -Destination (Join-Path $pdfDir "$job.pdf")
    }
} finally {
    Pop-Location
}
