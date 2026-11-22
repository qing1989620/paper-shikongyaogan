$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    New-Item -ItemType Directory -Path 'output/pdf','build/en','build/zh' -Force | Out-Null
    foreach ($language in @('en','zh')) {
        $sourceName = 'icip_' + $language + '_compact'
        $engine = if ($language -eq 'en') { '-pdf' } else { '-xelatex' }
        & latexmk $engine -silent -interaction=nonstopmode -halt-on-error ('-outdir=build/' + $language) ($sourceName + '.tex')
        if ($LASTEXITCODE -ne 0) { throw ('Compilation failed: ' + $language + '; inspect build/' + $language) }
        Copy-Item -LiteralPath ('build/' + $language + '/' + $sourceName + '.pdf') -Destination ('output/pdf/' + $sourceName + '.pdf')
        Write-Output ('Built output/pdf/' + $sourceName + '.pdf')
    }
} finally {
    Pop-Location
}
