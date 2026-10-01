use tauri::{Emitter, Manager, PhysicalPosition, PhysicalSize, WebviewWindow};
use tauri_plugin_global_shortcut::{
    Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState,
};

/// Window size in logical px. Small on purpose: while a confirm is shown the
/// window takes clicks, so keep the blocking region small.
const WIN_W: f64 = 760.0;
const WIN_H: f64 = 320.0;

/// true = clicks pass through to whatever is below; false = window takes clicks.
#[tauri::command]
fn set_clickthrough(window: WebviewWindow, ignore: bool) -> Result<(), String> {
    window.set_ignore_cursor_events(ignore).map_err(|e| e.to_string())
}

/// Bottom-center of the primary monitor's work area (so it sits above the taskbar).
fn place(win: &WebviewWindow) -> tauri::Result<()> {
    let Some(mon) = win.primary_monitor()? else {
        return Ok(());
    };
    let scale = mon.scale_factor();
    let w = (WIN_W * scale).round() as i32;
    let h = (WIN_H * scale).round() as i32;
    let area = mon.work_area();
    let x = area.position.x + (area.size.width as i32 - w) / 2;
    let y = area.position.y + area.size.height as i32 - h;
    win.set_size(PhysicalSize::new(w as u32, h as u32))?;
    win.set_position(PhysicalPosition::new(x, y))?;
    Ok(())
}

pub fn run() {
    let hotkey = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::ALT), Code::KeyJ);

    tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(move |app, shortcut, event| {
                    if shortcut == &hotkey && event.state() == ShortcutState::Pressed {
                        // Frontend owns the WebSocket and sends {type:"activate"}.
                        let _ = app.emit("hotkey", ());
                    }
                })
                .build(),
        )
        .invoke_handler(tauri::generate_handler![set_clickthrough])
        .setup(move |app| {
            if let Some(win) = app.get_webview_window("main") {
                place(&win)?;
                win.set_ignore_cursor_events(true)?;
                win.show()?;
            }
            // Not fatal if another app already owns the hotkey.
            if let Err(e) = app.global_shortcut().register(hotkey) {
                eprintln!("could not register Ctrl+Alt+J: {e}");
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Jarvis orb");
}
