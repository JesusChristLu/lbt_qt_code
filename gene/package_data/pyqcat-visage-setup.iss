; inno : 6.2.0, 官方版本安装后请把ChineseSimplified.isl文件复制到安装目录的Languages目录下。
;软件基础信息配置
#define MyAppName "PyQCat-Visage"
#define MyAppVersion "beta-c-0.1.0.1"
#define MyAppPublisher "合肥本源量子计算科技有限责任公司"
#define MyAppURL "https://originqc.com.cn/"

; 待打包文件配置，MyAppExePath为nuitka打包后的exe文件所在的文件夹，MyAppOtherDataPath为二次打包所需的配置文件文件夹
#define MyAppExePath "C:\tools\projects\cx_freeze\pyqcat-visage\build\2023_0214_0947"
#define MyAppOtherDataPath "C:\tools\projects\cx_freeze\pyqcat-visage\package_data"
#define MyAppExeName "visage.exe"

[Setup]
; 注: AppId的值为单独标识该应用程序。
; 不要为其他安装程序使用相同的AppId值。
; (若要生成新的 GUID，可在菜单中点击 "工具|生成 GUID"。)
; nuitka
; AppId={{C00F0B6B-B8A5-4928-A803-C4700F9CB00E}
; cx_freeze
AppId={{CE4B3247-E78A-444C-B625-6E09B931C139}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName=D:\Soft\{#MyAppName}
DisableProgramGroupPage=yes

LicenseFile={#MyAppOtherDataPath}\license.txt
InfoBeforeFile={#MyAppOtherDataPath}\install_before_info.txt
InfoAfterFile={#MyAppOtherDataPath}\install_after_info.txt
; 以下行取消注释，以在非管理安装模式下运行（仅为当前用户安装）。
;PrivilegesRequired=lowest
;PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#MyAppOtherDataPath}
OutputBaseFilename=PyQCat-Visage_setup_{#MyAppVersion}
SetupIconFile={#MyAppOtherDataPath}\favicon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern

UninstallDisplayIcon={app}\{#MyAppExeName}
Uninstallable=yes
UninstallDisplayName={#MyAppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#MyAppExePath}\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#MyAppExePath}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; 注意: 不要在任何共享系统文件上使用“Flags: ignoreversion”

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

