import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import webbrowser
import sys

if getattr(sys, 'frozen', False):
    os.chdir(sys._MEIPASS)  # PyInstaller 打包后的临时目录

class ApplicationGUI:
    def __init__(self, master):
        self.master = master
        master.title("Dolphin System Manager")

        # 初始化变量
        self.rag_model_path = ""
        self.rag_knowledge_path = ""
        self.audio_model_path = ""  # 新增的变量
        self.audio_save_path = ""
        self.db_path = ""
        self.api_key = ""

        # 定义字体
        self.default_font = ("微软雅黑", 14)

        # 创建路径输入和选择按钮
        self.create_path_controls()
        self.create_button_controls()
        self.create_status_display()

        # 用于存储app.py的进程
        self.process = None

        # 读取之前的配置
        self.load_config()

    def create_path_controls(self):
        self.create_label_entry("RAG 模型路径：", 0)
        self.create_label_entry("RAG 知识路径：", 1)
        self.create_label_entry("音频模型路径：", 2)  # 新增的输入框
        self.create_label_entry("音频保存路径：", 3)
        self.create_label_entry("data.db 保存路径：", 4)
        
        self.api_key_entry = tk.Entry(
            self.master,
            width=50,
            font=self.default_font
        )
        self.api_key_entry.grid(
            row=5,
            column=1,
            padx=5,
            pady=5,
            sticky="w"
        )
        tk.Label(
            self.master,
            text="API Key：",
            font=self.default_font
        ).grid(
            row=5,
            column=0,
            padx=5,
            pady=5,
            sticky="e"
        )

    def create_label_entry(self, label_text, row):
        tk.Label(
            self.master,
            text=label_text,
            font=self.default_font
        ).grid(
            row=row,
            column=0,
            padx=5,
            pady=5,
            sticky="e"
        )

        entry = tk.Entry(
            self.master,
            width=50,
            font=self.default_font
        )
        entry.grid(
            row=row,
            column=1,
            padx=5,
            pady=5,
            sticky="w"
        )

        browse_button = tk.Button(
            self.master,
            text="浏览",
            font=self.default_font,
            command=lambda: self.on_browse_click(entry, label_text)
        )
        browse_button.grid(
            row=row,
            column=2,
            padx=5,
            pady=5,
            sticky="w"
        )

    def on_browse_click(self, entry, label_text):
        if label_text == "RAG 模型路径：":
            folder_path = filedialog.askdirectory()
            entry.delete(0, tk.END)
            entry.insert(0, folder_path)
            self.rag_model_path = folder_path
        elif label_text == "RAG 知识路径：":
            file_path = filedialog.askopenfilename()
            entry.delete(0, tk.END)
            entry.insert(0, file_path)
            self.rag_knowledge_path = file_path
        elif label_text == "音频模型路径：":  # 新增的处理
            folder_path = filedialog.askdirectory()
            entry.delete(0, tk.END)
            entry.insert(0, folder_path)
            self.audio_model_path = folder_path
        elif label_text == "音频保存路径：":
            folder_path = filedialog.askdirectory()
            entry.delete(0, tk.END)
            entry.insert(0, folder_path)
            self.audio_save_path = folder_path
        elif label_text == "data.db 保存路径：":
            folder_path = filedialog.askdirectory()
            entry.delete(0, tk.END)
            entry.insert(0, folder_path)
            self.db_path = folder_path

    def create_button_controls(self):
        button_frame = tk.Frame(self.master)
        button_frame.grid(row=6, column=0, columnspan=3, pady=10)

        # 启动按钮
        self.start_button = tk.Button(
            button_frame,
            text="启动",
            font=self.default_font,
            command=self.start_server,
            bg="#4CAF50",
            fg="white"
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        # 停止按钮
        self.stop_button = tk.Button(
            button_frame,
            text="停止",
            font=self.default_font,
            command=self.stop_server,
            bg="#f44336",
            fg="white"
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # 打开网页按钮
        self.open_button = tk.Button(
            button_frame,
            text="打开网页",
            font=self.default_font,
            command=self.open_webpage,
            bg="#2196F3",
            fg="white"
        )
        self.open_button.pack(side=tk.LEFT, padx=5)

    def create_status_display(self):
        status_frame = tk.Frame(self.master)
        status_frame.grid(row=7, column=0, columnspan=3, pady=10)

        tk.Label(
            status_frame,
            text="运行状态：",
            font=self.default_font
        ).pack(side=tk.LEFT)

        self.status_label = tk.Label(
            status_frame,
            text="未运行",
            font=self.default_font,
            fg="red"
        )
        self.status_label.pack(side=tk.LEFT, padx=5)

    def start_server(self):
        if self.process and self.process.poll() is None:
            messagebox.showwarning("提示", "app.py 已经在运行中！")
            return

        # 获取输入框的值
        self.rag_model_path = self.master.grid_slaves(0, 1)[0].get()
        self.rag_knowledge_path = self.master.grid_slaves(1, 1)[0].get()
        self.audio_model_path = self.master.grid_slaves(2, 1)[0].get()  # 新增的获取
        self.audio_save_path = self.master.grid_slaves(3, 1)[0].get()
        self.db_path = self.master.grid_slaves(4, 1)[0].get()
        self.api_key = self.api_key_entry.get()

        # 校验路径和API Key
        if not all([self.rag_model_path, self.rag_knowledge_path, self.audio_model_path, self.audio_save_path, self.db_path, self.api_key]):
            messagebox.showerror("错误", "请填写所有路径和API Key！")
            return

        # 修改app.py中的路径和变量
        self.update_config_file()

        # 启动app.py
        self.process = subprocess.Popen(
            ["python", os.path.join(os.getcwd(), "app.py")],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        self.status_label.config(text="运行中", fg="green")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.save_config()

    def stop_server(self):
        if self.process:
            try:
                self.process.terminate()
                self.process = None
                self.status_label.config(text="未运行", fg="red")
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
            except Exception as e:
                messagebox.showerror("错误", f"停止服务失败：{str(e)}")

    def open_webpage(self):
        webbrowser.open("http://127.0.0.1:5000")

    def update_config_file(self):
        # 将用户输入的路径和API Key写入配置文件
        with open("config.py", "w", encoding="utf-8") as f:
            f.write(f'RAG_MODEL_PATH = "{self.rag_model_path}"\n')
            f.write(f'RAG_KNOWLEDGE_PATH = "{self.rag_knowledge_path}"\n')
            f.write(f'AUDIO_MODEL_PATH = "{self.audio_model_path}"\n')  # 新增的写入
            f.write(f'AUDIO_SAVE_PATH = "{self.audio_save_path}"\n')
            f.write(f'DB_PATH = "{self.db_path}"\n')
            f.write(f'API_KEY = "{self.api_key}"\n')

    def save_config(self):
        with open("config.txt", "w") as f:
            f.write(self.rag_model_path + "\n")
            f.write(self.rag_knowledge_path + "\n")
            f.write(self.audio_model_path + "\n")  # 新增的保存
            f.write(self.audio_save_path + "\n")
            f.write(self.db_path + "\n")
            f.write(self.api_key + "\n")

    def load_config(self):
        if os.path.exists("config.txt"):
            with open("config.txt", "r") as f:
                lines = f.readlines()
                if len(lines) >= 6:  # 修改为6
                    self.rag_model_path = lines[0].strip()
                    self.rag_knowledge_path = lines[1].strip()
                    self.audio_model_path = lines[2].strip()  # 新增的加载
                    self.audio_save_path = lines[3].strip()
                    self.db_path = lines[4].strip()
                    self.api_key = lines[5].strip()

                    # 设置输入框的值
                    self.master.grid_slaves(0, 1)[0].delete(0, tk.END)
                    self.master.grid_slaves(0, 1)[0].insert(0, self.rag_model_path)
                    self.master.grid_slaves(1, 1)[0].delete(0, tk.END)
                    self.master.grid_slaves(1, 1)[0].insert(0, self.rag_knowledge_path)
                    self.master.grid_slaves(2, 1)[0].delete(0, tk.END)
                    self.master.grid_slaves(2, 1)[0].insert(0, self.audio_model_path)
                    self.master.grid_slaves(3, 1)[0].delete(0, tk.END)
                    self.master.grid_slaves(3, 1)[0].insert(0, self.audio_save_path)
                    self.master.grid_slaves(4, 1)[0].delete(0, tk.END)
                    self.master.grid_slaves(4, 1)[0].insert(0, self.db_path)
                    self.api_key_entry.delete(0, tk.END)
                    self.api_key_entry.insert(0, self.api_key)

if __name__ == "__main__":
    root = tk.Tk()
    app = ApplicationGUI(root)
    root.mainloop()