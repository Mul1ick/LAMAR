import { app, BrowserWindow, dialog, ipcMain } from "electron";
import path from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process"; // <-- Add this import
import fs from "fs"; // <-- ADD THIS IMPORT

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const isDev = !app.isPackaged;

let backendProcess = null;

function startBackend() {
  if (isDev) return; 

  const backendPath = path.join(process.resourcesPath, "doclamar-backend", "doclamar-backend");
  
  // NEW: Ensure macOS allows the file to be executed
  try {
    if (fs.existsSync(backendPath)) {
      fs.chmodSync(backendPath, '755');
    }
  } catch (err) {
    console.error("Failed to set permissions:", err);
  }

  backendProcess = spawn(backendPath, [], { 
    detached: false, 
    cwd: process.resourcesPath // <-- This tells Python to look here for the .env file!
  });

  backendProcess.stdout.on('data', (data) => console.log(`Backend: ${data}`));
  backendProcess.stderr.on('data', (data) => console.error(`Backend Error: ${data}`));
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1920,
    height: 1080,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, "preload.cjs"),
    },
  });

  if (isDev) {
    // Load Vite dev server
    win.loadURL("http://localhost:5173");
    win.webContents.openDevTools();
  } else {
    // Load built production files
    win.loadFile(path.join(__dirname, "dist", "index.html"));
  }
}

app.whenReady().then(() => {
  startBackend();
  createWindow();

  // Handle file dialog
  ipcMain.handle('dialog:openDirectory', async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openDirectory']
    });
    return result;
  });
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on('will-quit', () => {
  if (backendProcess) {
    backendProcess.kill();
  }
});
