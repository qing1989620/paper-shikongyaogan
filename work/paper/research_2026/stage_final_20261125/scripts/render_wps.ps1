param(
    [Parameter(Mandatory=$true)][string]$DeckPath,
    [Parameter(Mandatory=$true)][string]$Destination,
    [string]$PdfPath,
    [int]$Width=2000
)

$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $DeckPath).Path
$destinationDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Destination)
New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null
$wps = New-Object -ComObject KWPP.Application
$deck = $null
try {
    $deck = $wps.Presentations.Open($source,-1,0,0)
    if ($deck.FullName -ne $source) { throw "Unexpected presentation opened: $($deck.FullName)" }
    $height = [int][math]::Round($Width * $deck.PageSetup.SlideHeight / $deck.PageSetup.SlideWidth)
    $pages = @()
    for ($i=1; $i -le $deck.Slides.Count; $i++) {
        $slide = $deck.Slides.Item($i)
        $png = Join-Path $destinationDir ('slide_{0:00}.png' -f $i)
        $slide.Export($png,'PNG',$Width,$height)
        if (!(Test-Path -LiteralPath $png -PathType Leaf)) { throw "PNG not exported: $png" }
        $overflow = @()
        for ($j=1; $j -le $slide.Shapes.Count; $j++) {
            $shape = $slide.Shapes.Item($j)
            if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
                $text = $shape.TextFrame.TextRange
                if ($text.BoundWidth -gt ($shape.Width + 1.1) -or $text.BoundHeight -gt ($shape.Height + 2.1)) {
                    $overflow += [pscustomobject]@{
                        name=$shape.Name; width=$shape.Width; height=$shape.Height
                        textWidth=$text.BoundWidth; textHeight=$text.BoundHeight
                    }
                }
            }
        }
        $pages += [pscustomobject]@{slide=$i; png=$png; objects=$slide.Shapes.Count; overflow=$overflow}
    }
    if ($PdfPath) {
        $pdf = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($PdfPath)
        $deck.SaveAs($pdf,32)
        if (!(Test-Path -LiteralPath $pdf -PathType Leaf)) { throw "PDF not exported: $pdf" }
    }
    $renderer = @(Get-Process -Name wpp -ErrorAction SilentlyContinue | Where-Object Path | Select-Object -ExpandProperty Path)
    [pscustomobject]@{
        app='WPS Presentation'; version=$wps.Version; opened=$deck.FullName
        renderer=$renderer; renderedSize=@($Width,$height); slides=$pages
    } | ConvertTo-Json -Depth 7
} finally {
    if ($deck) { $deck.Close() }
    [Runtime.InteropServices.Marshal]::ReleaseComObject($wps) | Out-Null
}
