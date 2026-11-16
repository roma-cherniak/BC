# Close the AutoDRIVE simulator and open the build of one track, then click its "Connect" button.
#   powershell -File night\open_sim.ps1 compete      (or practice;  "close" only closes the sim)
# The user allowed this on 2026-11-12 ("both yes": close / open the builds, move the mouse for the click).
# The clicks are places measured in the 604 x 394 window the sim first opened with (the menu button, then
# "Disconnected"); they are scaled to the size the window has now (12 Nov: at 480 x 284 the fixed places missed).
# A driver must already be listening on port 4567, or the click connects to nothing.
param([Parameter(Mandatory = $true)][ValidateSet("practice", "compete", "close")][string]$Track)

$Name = "AutoDRIVE Simulator"
$Folder = "C:\Users\1\Desktop\BP\autodrive_simulator_${Track}_windows\autodrive_simulator"
$Menu = @(32, 52)          # the three lines, top left
$Connect = @(32, 170)      # "Disconnected" in the opened menu

Add-Type @'
using System; using System.Runtime.InteropServices;
public class SimWindow {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
  public struct POINT { public int X, Y; }
  [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr h, int x, int y, int w, int ht, bool repaint);
  [DllImport("user32.dll")] public static extern void keybd_event(byte k, byte s, uint f, UIntPtr e);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, uint d, UIntPtr e);
  public struct RECT { public int L, T, R, B; } }
'@
[SimWindow]::SetProcessDPIAware() | Out-Null

function Click-In($Process, $Place) {
    # bring the window to the front (Windows only allows it after a key press: Alt), then click
    [SimWindow]::keybd_event(0x12, 0, 0, [UIntPtr]::Zero)
    [SimWindow]::ShowWindow($Process.MainWindowHandle, 9) | Out-Null
    [SimWindow]::SetForegroundWindow($Process.MainWindowHandle) | Out-Null
    [SimWindow]::keybd_event(0x12, 0, 2, [UIntPtr]::Zero)
    Start-Sleep -Milliseconds 800
    # 14 Nov: measured from the inside of the window (client area, 604 x 355 when the places were measured), so the
    # height of the title bar (39 px on 12 Nov, ~72 px at 125 % display scaling) no longer matters
    $c = New-Object SimWindow+RECT
    [SimWindow]::GetClientRect($Process.MainWindowHandle, [ref]$c) | Out-Null
    $o = New-Object SimWindow+POINT
    [SimWindow]::ClientToScreen($Process.MainWindowHandle, [ref]$o) | Out-Null
    $x = [int]($Place[0] * $c.R / 604)
    $y = [int](($Place[1] - 39) * $c.B / 355)
    [SimWindow]::SetCursorPos($o.X + $x, $o.Y + $y) | Out-Null
    Start-Sleep -Milliseconds 200
    [SimWindow]::mouse_event(2, 0, 0, 0, [UIntPtr]::Zero); Start-Sleep -Milliseconds 80
    [SimWindow]::mouse_event(4, 0, 0, 0, [UIntPtr]::Zero)
    Start-Sleep -Milliseconds 1500
}

Get-Process $Name -ErrorAction SilentlyContinue | Stop-Process -Force -Confirm:$false
Start-Sleep -Seconds 3
if ($Track -eq "close") { "sim closed"; exit 0 }
if (-not (Test-Path "$Folder\$Name.exe")) { "no sim build in $Folder"; exit 1 }
Start-Process "$Folder\$Name.exe" -WorkingDirectory $Folder
Start-Sleep -Seconds 15
$p = Get-Process $Name -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $p -or $p.MainWindowHandle -eq 0) { "the sim did not open"; exit 1 }
# 14 Nov: the sim may open almost off the screen (x 1472 of 1536): the clicks then miss. Move it to the top left first.
$r0 = New-Object SimWindow+RECT
[SimWindow]::GetWindowRect($p.MainWindowHandle, [ref]$r0) | Out-Null
[SimWindow]::MoveWindow($p.MainWindowHandle, 20, 20, $r0.R - $r0.L, $r0.B - $r0.T, $true) | Out-Null
Start-Sleep -Seconds 1
Click-In $p $Menu
Click-In $p $Connect
"sim open: $Track, connect clicked"
