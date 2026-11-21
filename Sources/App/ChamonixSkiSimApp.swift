import AppKit
import SwiftUI

struct RootContentView: View {
    @ObservedObject var appState: AppState

    var body: some View {
        ZStack {
            MetalViewportView(appState: appState)
                .ignoresSafeArea()
            AlpineHUDView(appState: appState)
        }
        .background(Color(red: 10/255, green: 16/255, blue: 26/255))
        .frame(minWidth: 800, minHeight: 540)
    }
}

@main
enum ChamonixSkiSimMain {
    @MainActor
    static func main() {
        let cli = CLIOptions.parse(CommandLine.arguments)
        let app = NSApplication.shared
        let delegate = AppDelegate(cli: cli)
        app.delegate = delegate
        app.setActivationPolicy(.regular)
        app.run()
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let appState: AppState
    private var window: NSWindow?

    init(cli: CLIOptions) {
        self.appState = AppState(cli: cli)
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let w = appState.cli.windowWidth
        let h = appState.cli.windowHeight
        let rect = NSRect(x: 0, y: 0, width: w, height: h)
        let win = NSWindow(
            contentRect: rect,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        win.title = "CHAMONIX · Mont Blanc Off-Piste Freeride Simulator"
        win.backgroundColor = NSColor(red: 10/255, green: 16/255, blue: 26/255, alpha: 1.0)
        win.contentMinSize = NSSize(width: 800, height: 540)
        win.contentView = NSHostingView(rootView: RootContentView(appState: appState))
        win.center()
        win.makeKeyAndOrderFront(nil)
        self.window = win

        NSApp.activate(ignoringOtherApps: true)
    }

    func application(_ sender: NSApplication, openFile filename: String) -> Bool {
        false
    }

    func application(_ sender: NSApplication, openFiles filenames: [String]) {
        sender.reply(toOpenOrPrint: .cancel)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
