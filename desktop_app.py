import customtkinter
import pandas as pd
from tkinter import filedialog
import os 
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.cluster import KMeans
import threading

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

from fuzzy_logic_engine import get_initial_system_state, get_optimized_system_state

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Аналітичний центр енергосистеми")
        self.geometry("1200x800") 
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)
        self.tab_interactive = self.tabview.add("Інтерактивний аналіз")
        self.tab_report = self.tabview.add("Аналіз файлу та звіт")
        self.tab_rules = self.tabview.add("Перегляд правил")
        self.tab_viz = self.tabview.add("3D Візуалізація")
        self.tab_cluster = self.tabview.add("Кластеризація")
        self.create_interactive_tab(self.tab_interactive)
        self.create_report_tab(self.tab_report)
        self.create_rules_tab(self.tab_rules)
        self.create_viz_tab(self.tab_viz)
        self.create_cluster_tab(self.tab_cluster)
        self.report_data = []
        self._viz_cache = {}  
        self._viz_worker_stop_event = threading.Event()
        self.report_summary = None

    def create_interactive_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=3)
        tab.grid_rowconfigure(0, weight=1)
        
        controls_frame = customtkinter.CTkFrame(tab)
        controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        title_label = customtkinter.CTkLabel(controls_frame, text="Ручний аналіз", font=("Arial", 18))
        title_label.pack(padx=20, pady=10)
        
        self.consumption_label = customtkinter.CTkLabel(controls_frame, text="Поточне споживання (%): 50")
        self.consumption_label.pack(pady=(10,0))
        self.consumption_slider = customtkinter.CTkSlider(controls_frame, from_=0, to=100, number_of_steps=100)
        self.consumption_slider.set(50)
        self.consumption_slider.pack(padx=20, pady=10)
        
        self.frequency_label = customtkinter.CTkLabel(controls_frame, text="Відхилення частоти (Гц): 0.00")
        self.frequency_label.pack(pady=(10,0))
        self.frequency_slider = customtkinter.CTkSlider(controls_frame, from_=-0.5, to=0.5, number_of_steps=100)
        self.frequency_slider.set(0)
        self.frequency_slider.pack(padx=20, pady=10)
        
        result_frame = customtkinter.CTkFrame(controls_frame)
        result_frame.pack(padx=20, pady=20, fill="both", expand=True)
        self.initial_result_label = customtkinter.CTkLabel(result_frame, text="", font=("Arial", 20))
        self.initial_result_label.pack(pady=(10, 0))
        self.optimized_result_label = customtkinter.CTkLabel(result_frame, text="", font=("Arial", 20, "bold"))
        self.optimized_result_label.pack(pady=(20, 0))

        plot_frame = customtkinter.CTkFrame(tab)
        plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.fig, self.axs = plt.subplots(3, 2, figsize=(10, 8), facecolor='#2B2B2B')
        plt.style.use('dark_background')
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=customtkinter.TOP, fill=customtkinter.BOTH, expand=1)
        
        self.consumption_slider.configure(command=self.update_interactive_values)
        self.frequency_slider.configure(command=self.update_interactive_values)

        self.update_interactive_values()

    def update_interactive_values(self, _=None): 
        consumption_val = self.consumption_slider.get()
        frequency_val = self.frequency_slider.get()

        self.consumption_label.configure(text=f"Поточне споживання (%): {consumption_val:.0f}")
        self.frequency_label.configure(text=f"Відхилення частоти (Гц): {frequency_val:.2f}")
        
        initial_state = get_initial_system_state(consumption_val, frequency_val)
        optimized_state = get_optimized_system_state(consumption_val, frequency_val)
        
        initial_text, initial_color = self._get_state_display(initial_state)
        self.initial_result_label.configure(text=f"Початкова: {initial_text} ({initial_state:.2f}/10)", text_color=initial_color)
        
        optimized_text, optimized_color = self._get_state_display(optimized_state)
        self.optimized_result_label.configure(text=f"Оптимізована: {optimized_text} ({optimized_state:.2f}/10)", text_color=optimized_color)

        self._update_plots(consumption_val, frequency_val, initial_state, optimized_state)
    
    def create_report_tab(self, tab):
        top_frame = customtkinter.CTkFrame(tab); top_frame.pack(padx=10, pady=10, fill="x")
        self.button_load = customtkinter.CTkButton(top_frame, text="Проаналізувати 'power_load_hourly.csv'", command=self.process_csv_file); self.button_load.pack(side="left", padx=10, pady=10)
        self.button_save = customtkinter.CTkButton(top_frame, text="Зберегти звіт", command=self.save_report, state="disabled"); self.button_save.pack(side="left", padx=10, pady=10)
        self.analysis_limit_label = customtkinter.CTkLabel(top_frame, text="Макс рядків: 1000"); self.analysis_limit_label.pack(side="left", padx=(10,5))
        self.analysis_limit_slider = customtkinter.CTkSlider(top_frame, from_=100, to=10000, number_of_steps=99, command=self._update_limit_label); self.analysis_limit_slider.set(1000); self.analysis_limit_slider.pack(side="left", padx=5)
        self.button_analysis = customtkinter.CTkButton(top_frame, text="Показати аналіз ефективності", command=self.show_analysis_summary); self.button_analysis.pack(side="left", padx=10, pady=10)
        self.progress_label = customtkinter.CTkLabel(top_frame, text=""); self.progress_label.pack(side="left", padx=10, pady=10)
        self.progressbar = customtkinter.CTkProgressBar(top_frame, orientation="horizontal"); self.progressbar.set(0)
        self.report_frame = customtkinter.CTkScrollableFrame(tab, label_text="Звіт по аналізу даних"); self.report_frame.pack(padx=10, pady=10, fill="both", expand=True)
        self.create_table_header()
    def create_rules_tab(self, tab):
        
        rules_frame = customtkinter.CTkFrame(tab); rules_frame.pack(padx=12, pady=12, fill="both", expand=True)
        left = customtkinter.CTkFrame(rules_frame); left.pack(side="left", fill="both", expand=True, padx=(0,6))
        right = customtkinter.CTkFrame(rules_frame); right.pack(side="left", fill="both", expand=True, padx=(6,0))

        
        self.initial_rules_text = (
            "1. ЯКЩО споживання 'Normal' І частота 'Stable' ТО стан 'Stable'\n"
            "2. ЯКЩО споживання 'High' І частота 'Negative_Low' ТО стан 'Warning'\n"
            "3. ЯКЩО споживання 'Critical' І частота 'Negative_High' ТО стан 'Danger'\n"
            "4. ЯКЩО споживання 'Low' І частота 'Positive' ТО стан 'Warning'\n"
            "5. ЯКЩО споживання 'High' АБО 'Critical' ТО стан 'Danger'\n"
            "6. ЯКЩО споживання 'Low' ТО стан 'Stable'"
        )
        self.optimized_rules_text = (
            "--- Стабільні сценарії ---\n"
            "1. ЯКЩО споживання 'Low' І частота 'Stable' ТО стан 'Stable'\n"
            "2. ЯКЩО споживання 'Low' І частота 'Negative_Low' ТО стан 'Stable'\n"
            "3. ЯКЩО споживання 'Low' І частота 'Positive' ТО стан 'Stable'\n"
            "4. ЯКЩО споживання 'Normal' І частота 'Stable' ТО стан 'Stable'\n"
            "--- Сценарії попередження ---\n"
            "5. ЯКЩО споживання 'Low' І частота 'Negative_High' ТО стан 'Warning'\n"
            "6. ЯКЩО споживання 'Normal' І частота 'Negative_Low' ТО стан 'Warning'\n"
            "7. ЯКЩО споживання 'Normal' І частота 'Positive' ТО стан 'Warning'\n"
            "8. ЯКЩО споживання 'High' І частота 'Stable' ТО стан 'Warning'\n"
            "9. ЯКЩО споживання 'High' І частота 'Negative_Low' ТО стан 'Warning'\n"
            "10. ЯКЩО споживання 'High' І частота 'Positive' ТО стан 'Warning'\n"
            "11. ЯКЩО споживання 'Critical' І частота 'Stable' ТО стан 'Warning'\n"
            "--- Небезпечні сценарії ---\n"
            "12. ЯКЩО споживання 'Critical' І частота 'Positive' ТО стан 'Danger'\n"
            "13. ЯКЩО споживання 'Normal' І частота 'Negative_High' ТО стан 'Danger'\n"
            "14. ЯКЩО споживання 'High' І частота 'Negative_High' ТО стан 'Danger'\n"
            "15. ЯКЩО споживання 'Critical' І частота 'Negative_Low' ТО стан 'Danger'\n"
            "16. ЯКЩО споживання 'Critical' І частота 'Negative_High' ТО стан 'Danger'\n"
            "--- Загальні правила безпеки ---\n"
            "17. ЯКЩО споживання 'Critical' ТО стан 'Danger'\n"
            "18. ЯКЩО частота 'Negative_High' ТО стан 'Danger'\n"
            "19. ЯКЩО споживання 'High' ТО стан 'Warning'\n"
            "20. ЯКЩО споживання 'Low' ТО стан 'Stable'"
        )

        initial_label = customtkinter.CTkLabel(left, text="Початкова база правил (нередагована)", font=("Arial", 16, "bold"))
        initial_label.pack(anchor="w", pady=(6,4))
        self.initial_textbox = customtkinter.CTkTextbox(left, height=360, font=("Arial", 12), wrap="word")
        self.initial_textbox.insert("1.0", self.initial_rules_text); self.initial_textbox.configure(state="disabled")
        self.initial_textbox.pack(fill="both", expand=True, padx=6, pady=(0,6))

        optimized_label = customtkinter.CTkLabel(right, text="Оптимізована база правил (редагуйте при потребі)", font=("Arial", 16, "bold"))
        optimized_label.pack(anchor="w", pady=(6,4))
        self.optimized_textbox = customtkinter.CTkTextbox(right, height=300, font=("Arial", 12), wrap="word")
        self.optimized_textbox.insert("1.0", self.optimized_rules_text)
        self.optimized_textbox.pack(fill="both", expand=True, padx=6, pady=(0,6))

        btn_frame = customtkinter.CTkFrame(right)
        btn_frame.pack(fill="x", padx=6, pady=(0,6))
        btn_save = customtkinter.CTkButton(btn_frame, text="Зберегти в файл", command=self._save_rules_file); btn_save.pack(side="left", padx=6)
        btn_load = customtkinter.CTkButton(btn_frame, text="Завантажити з файлу", command=self._load_rules_file); btn_load.pack(side="left", padx=6)
        btn_reset = customtkinter.CTkButton(btn_frame, text="Скинути до дефолту", command=self._reset_rules_to_default); btn_reset.pack(side="left", padx=6)
        btn_copy = customtkinter.CTkButton(btn_frame, text="Копіювати", command=self._copy_rules_to_clipboard); btn_copy.pack(side="left", padx=6)

    def _save_rules_file(self):
        txt = self.optimized_textbox.get("1.0", "end").rstrip()
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files","*.txt"),("All","*.*")])
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as f: f.write(txt)
        except Exception as e:
            print(f"Помилка збереження файлу правил: {e}")

    def _load_rules_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text Files","*.txt"),("All","*.*")])
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.optimized_textbox.delete("1.0", "end"); self.optimized_textbox.insert("1.0", content)
        except Exception as e:
            print(f"Помилка завантаження файлу правил: {e}")

    def _reset_rules_to_default(self):
        self.optimized_textbox.delete("1.0", "end"); self.optimized_textbox.insert("1.0", self.optimized_rules_text)

    def _copy_rules_to_clipboard(self):
        txt = self.optimized_textbox.get("1.0", "end").rstrip()
        try:
            self.clipboard_clear(); self.clipboard_append(txt)
        except Exception:
            pass
    def create_viz_tab(self, tab):
        top_frame = customtkinter.CTkFrame(tab); top_frame.pack(padx=10, pady=10, fill="x")
        self.viz_button = customtkinter.CTkButton(top_frame, text="Побудувати поверхні відгуку", command=self.generate_and_plot_surfaces); self.viz_button.pack(side="left", padx=10, pady=10)
        self.viz_label = customtkinter.CTkLabel(top_frame, text="Натисніть кнопку, щоб почати розрахунок (може зайняти кілька секунд)"); self.viz_label.pack(side="left", padx=10, pady=10)
        
        self.viz_res_label = customtkinter.CTkLabel(top_frame, text="Роздільність сітки:"); self.viz_res_label.pack(side="left", padx=(10,5))
        self.viz_resolution_slider = customtkinter.CTkSlider(top_frame, from_=10, to=60, number_of_steps=50); self.viz_resolution_slider.set(30); self.viz_resolution_slider.pack(side="left", padx=5)
        self.viz_cancel = customtkinter.CTkButton(top_frame, text="Скасувати", command=self._viz_cancel, state="disabled"); self.viz_cancel.pack(side="left", padx=10)
        
        self.viz_progressbar = customtkinter.CTkProgressBar(top_frame, orientation="horizontal")
        self.viz_progressbar.set(0)
        plot_frame = customtkinter.CTkFrame(tab); plot_frame.pack(padx=10, pady=10, fill="both", expand=True)
        self.fig_3d = plt.figure(figsize=(12, 6), facecolor='#2B2B2B'); self.ax1_3d = self.fig_3d.add_subplot(1, 2, 1, projection='3d'); self.ax2_3d = self.fig_3d.add_subplot(1, 2, 2, projection='3d'); plt.style.use('dark_background')
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=plot_frame); self.canvas_3d.get_tk_widget().pack(side=customtkinter.TOP, fill=customtkinter.BOTH, expand=1)

    def create_cluster_tab(self, tab):
        top_frame = customtkinter.CTkFrame(tab); top_frame.pack(padx=10, pady=10, fill="x")
        self.cluster_button = customtkinter.CTkButton(top_frame, text="Виконати кластеризацію даних", command=self.perform_clustering); self.cluster_button.pack(side="left", padx=10, pady=10)
        self.cluster_label = customtkinter.CTkLabel(top_frame, text="Кількість кластерів (K):"); self.cluster_label.pack(side="left", padx=(20, 5), pady=10)
        self.cluster_k_slider = customtkinter.CTkSlider(top_frame, from_=2, to=8, number_of_steps=6); self.cluster_k_slider.set(4); self.cluster_k_slider.pack(side="left", padx=5, pady=10)
        plot_frame = customtkinter.CTkFrame(tab); plot_frame.pack(padx=10, pady=10, fill="both", expand=True)
        self.fig_cluster = plt.figure(figsize=(10, 8), facecolor='#2B2B2B'); self.ax_cluster = self.fig_cluster.add_subplot(1, 1, 1); plt.style.use('dark_background')
        self.canvas_cluster = FigureCanvasTkAgg(self.fig_cluster, master=plot_frame); self.canvas_cluster.get_tk_widget().pack(side=customtkinter.TOP, fill=customtkinter.BOTH, expand=1)
    def perform_clustering(self):
        filepath = 'power_load_hourly.csv'
        try:
            df = pd.read_csv(filepath); df.columns = df.columns.str.lower()
            data_for_clustering = df[['навантаження_мвт', 'температура_с']].dropna()
        except FileNotFoundError:
            self.ax_cluster.clear(); self.ax_cluster.text(0.5, 0.5, f"ПОМИЛКА: Файл '{filepath}' не знайдено!", color='red', ha='center', va='center'); self.canvas_cluster.draw()
            return
        except KeyError:
            self.ax_cluster.clear(); self.ax_cluster.text(0.5, 0.5, "ПОМИЛКА: У файлі відсутні стовпці 'навантаження_мвт' або 'температура_с'", color='red', ha='center', va='center'); self.canvas_cluster.draw()
            return
        k = int(self.cluster_k_slider.get())
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(data_for_clustering)
        self.ax_cluster.clear()
        scatter = self.ax_cluster.scatter(data_for_clustering['температура_с'], data_for_clustering['навантаження_мвт'], c=clusters, cmap='viridis', alpha=0.6)
        centers = kmeans.cluster_centers_
        self.ax_cluster.scatter(centers[:, 1], centers[:, 0], c='red', s=200, alpha=0.9, marker='X')
        self.ax_cluster.set_title(f'Кластеризація режимів роботи (K={k})', color='white'); self.ax_cluster.set_xlabel('Температура (°C)', color='white'); self.ax_cluster.set_ylabel('Навантаження (МВт)', color='white'); self.ax_cluster.tick_params(colors='white')
        legend1 = self.ax_cluster.legend(*scatter.legend_elements(), title="Кластери"); self.ax_cluster.add_artist(legend1)
        self.canvas_cluster.draw()
    def generate_and_plot_surfaces(self):
        cons_steps = int(self.viz_resolution_slider.get())
        freq_steps = cons_steps
        cache_key = (cons_steps, freq_steps)
        if cache_key in self._viz_cache:
            x, y, z_initial, z_optimized = self._viz_cache[cache_key]
            self.viz_label.configure(text=f"Відновлено з кеша (розд. {cons_steps}x{freq_steps})")
            self._plot_surfaces(x, y, z_initial, z_optimized)
            return

        self.viz_label.configure(text="Йде розрахунок..."); self.viz_button.configure(state="disabled"); self.update_idletasks()
        try:
            self.viz_progressbar.pack(side="left", padx=10, pady=10, fill="x", expand=True)
            self.viz_progressbar.set(0)
        except Exception:
            pass
        self._viz_worker_stop_event.clear()
        self.viz_cancel.configure(state="normal")
        worker = threading.Thread(target=self._generate_surfaces_worker, args=(cons_steps, freq_steps), daemon=True)
        worker.start()

    def _viz_cancel(self):
        self._viz_worker_stop_event.set()
        self.viz_label.configure(text="Скасовано... очікуйте")
        self.viz_cancel.configure(state="disabled")

    def _generate_surfaces_worker(self, cons_steps, freq_steps):
        try:
            cons_range = np.linspace(0, 100, cons_steps)
            freq_range = np.linspace(-0.5, 0.5, freq_steps)
            x, y = np.meshgrid(cons_range, freq_range)
            z_initial = np.zeros_like(x)
            z_optimized = np.zeros_like(x)
            total = x.size
            processed = 0
            update_step = max(1, total // 100)  
            for i in range(x.shape[0]):
                if self._viz_worker_stop_event.is_set():
                    self.after(0, lambda: (self.viz_label.configure(text="Розрахунок скасовано"), self.viz_progressbar.pack_forget(), self.viz_button.configure(state="normal"), self.viz_cancel.configure(state="disabled")))
                    return
                for j in range(x.shape[1]):
                    cv = float(x[i, j]); fv = float(y[i, j])
                    z_initial[i, j] = get_initial_system_state(cv, fv)
                    z_optimized[i, j] = get_optimized_system_state(cv, fv)
                    processed += 1
                    if processed % update_step == 0 or processed == total:
                        progress = processed / total
                        self.after(0, lambda p=progress: (self.viz_progressbar.set(p), self.viz_label.configure(text=f"Розрахунок... {int(p*100)}%")))
            self._viz_cache[(cons_steps, freq_steps)] = (x, y, z_initial, z_optimized)
            self.after(0, lambda: self._plot_surfaces(x, y, z_initial, z_optimized))
        except Exception as e:
            self.after(0, lambda: (self.viz_label.configure(text=f"Помилка: {e}"), self.viz_button.configure(state="normal"), self.viz_progressbar.pack_forget(), self.viz_cancel.configure(state="disabled")))

    def _plot_surfaces(self, x, y, z_initial, z_optimized):
        try:
            self.ax1_3d.clear(); self.ax2_3d.clear()
            self.ax1_3d.plot_surface(x, y, z_initial, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)
            self.ax1_3d.set_title('Початкова система'); self.ax1_3d.set_xlabel('Споживання (%)'); self.ax1_3d.set_ylabel('Відхилення частоти (Гц)'); self.ax1_3d.set_zlabel('Рівень загрози')
            self.ax2_3d.plot_surface(x, y, z_optimized, rstride=1, cstride=1, cmap='plasma', linewidth=0.4, antialiased=True)
            self.ax2_3d.set_title('Оптимізована система'); self.ax2_3d.set_xlabel('Споживання (%)'); self.ax2_3d.set_ylabel('Відхилення частоти (Гц)'); self.ax2_3d.set_zlabel('Рівень загрози')
            self.canvas_3d.draw()
            self.viz_label.configure(text="Розрахунок завершено.")
        finally:
            try: self.viz_progressbar.pack_forget()
            except Exception: pass
            self.viz_button.configure(state="normal")
            self.viz_cancel.configure(state="disabled")

    def _update_plots(self, cons_val, freq_val, initial_state_val, optimized_state_val):
        params_initial = {'consumption': {'name': 'Споживання', 'universe': np.arange(0, 101, 1), 'terms': {'Низьке': [0, 0, 40], 'Нормальне': [30, 50, 70], 'Високе': [60, 75, 90], 'Критичне': [85, 100, 100]}}, 'frequency': {'name': 'Відх. частоти', 'universe': np.arange(-0.5, 0.51, 0.01), 'terms': {'Значне пад.': [-0.5, -0.5, -0.2], 'Незначне пад.': [-0.3, -0.15, 0], 'Стабільна': [-0.1, 0, 0.1], 'Підвищена': [0.05, 0.5, 0.5]}}, 'state': {'name': 'Стан системи', 'universe': np.arange(0, 11, 1), 'terms': {'Стабільний': [0, 1.5, 3], 'Попередження': [2, 4.5, 7], 'Небезпека': [6, 8, 10]}}}
        self._plot_variable_manual(self.axs[0, 0], params_initial['consumption'], 'Початкова система', cons_val); self._plot_variable_manual(self.axs[1, 0], params_initial['frequency'], '', freq_val); self._plot_variable_manual(self.axs[2, 0], params_initial['state'], '', initial_state_val)
        params_optimized = {'consumption': {'name': 'Споживання', 'universe': np.arange(0, 101, 1), 'terms': {'Низьке': [0, 0, 30], 'Нормальне': [25, 50, 75], 'Високе': [70, 80, 90], 'Критичне': [85, 100, 100]}}, 'frequency': {'name': 'Відх. частоти', 'universe': np.arange(-0.5, 0.51, 0.01), 'terms': {'Значне пад.': [-0.5, -0.5, -0.2], 'Незначне пад.': [-0.25, -0.1, 0], 'Стабільна': [-0.05, 0, 0.05], 'Підвищена': [0.05, 0.5, 0.5]}}, 'state': {'name': 'Стан системи', 'universe': np.arange(0, 11, 1), 'terms': {'Стабільний': [0, 1.5, 3], 'Попередження': [2.5, 4.5, 7], 'Небезпека': [6.5, 8, 10]}}}
        self._plot_variable_manual(self.axs[0, 1], params_optimized['consumption'], 'Оптимізована система', cons_val); self._plot_variable_manual(self.axs[1, 1], params_optimized['frequency'], '', freq_val); self._plot_variable_manual(self.axs[2, 1], params_optimized['state'], '', optimized_state_val)
        self.fig.tight_layout(); self.canvas.draw()
    def _plot_variable_manual(self, ax, variable_params, title, current_value=None):
        ax.clear(); ax.set_title(title, color='white'); universe = variable_params['universe']
        for label, params in variable_params['terms'].items():
            mf = fuzz.trimf(universe, params); ax.plot(universe, mf, label=label)
        if current_value is not None: ax.axvline(current_value, color='r', linestyle='--')
        ax.legend(loc='upper right'); ax.tick_params(colors='white'); ax.yaxis.label.set_color('white'); ax.xaxis.label.set_color('white'); ax.set_ylabel(variable_params['name'], color='white')
    def create_table_header(self):
        for widget in self.report_frame.winfo_children(): widget.destroy()
        header_frame = customtkinter.CTkFrame(self.report_frame, fg_color="gray20"); header_frame.pack(fill="x", expand=True)
        headers = ["Час", "Навантаження (%)", "Частота (Гц)", "Оцінка (стара)", "Вердикт (старий)", "Оцінка (нова)", "Вердикт (новий)"]
        for i, header in enumerate(headers):
            label = customtkinter.CTkLabel(header_frame, text=header, font=("Arial", 12, "bold")); label.grid(row=0, column=i, padx=5, pady=5, sticky="ew")
        for col in range(len(headers)):
            header_frame.grid_columnconfigure(col, weight=1)

    def process_csv_file(self):
        filepath = 'power_load_hourly.csv'; self.create_table_header(); self.report_data.clear()
        stats_initial = {"Стабільний":0, "Попередження":0, "Небезпека":0}
        stats_optimized = {"Стабільний":0, "Попередження":0, "Небезпека":0}
        sum_initial = 0.0; sum_optimized = 0.0
        improvements = 0; regressions = 0; valid_rows = 0

        try:
            self.analysis_limit_slider.configure(state="disabled")
            self.analysis_limit_label.configure(text=self.analysis_limit_label.cget("text"))
        except Exception:
            pass

        self.button_load.configure(state="disabled"); self.button_save.configure(state="disabled"); self.progress_label.configure(text="Обробка..."); self.progressbar.pack(side="left", padx=10, pady=10, fill="x", expand=True); self.update_idletasks()
        try:
            df = pd.read_csv(filepath); df.columns = df.columns.str.lower()
        except FileNotFoundError:
            self.create_table_header(); error_label = customtkinter.CTkLabel(self.report_frame, text=f"ПОМИЛКА: Файл '{filepath}' не знайдено!", text_color="red", font=("Arial", 14)); error_label.pack(pady=20)
            self.button_load.configure(state="normal"); self.progressbar.pack_forget(); self.progress_label.configure(text="")
            try: self.analysis_limit_slider.configure(state="normal")
            except Exception: pass
            return

        total_rows = len(df)
        try:
            max_rows = int(self.analysis_limit_slider.get())
        except Exception:
            max_rows = total_rows
        total_to_process = min(total_rows, max_rows)

        processed_attempts = 0
        update_step = max(1, total_to_process // 100)

        for index, row in df.iterrows():
            if valid_rows >= total_to_process:
                break
            try:
                if row.get('потужність_мвт', 0) == 0:
                    continue
                consumption_percent = (row['навантаження_мвт'] / row['потужність_мвт']) * 100
                freq_dev = 0.0; load_ratio = consumption_percent / 100
                if load_ratio > 0.75: freq_dev = -0.1 * (load_ratio - 0.75) * 4
                elif load_ratio < 0.30: freq_dev = 0.05 * (0.30 - load_ratio) * 3.33
                initial_state = get_initial_system_state(consumption_percent, freq_dev); optimized_state = get_optimized_system_state(consumption_percent, freq_dev)
                initial_verdict, _ = self._get_state_display(initial_state); optimized_verdict, _ = self._get_state_display(optimized_state)
                report_row = { "Час": row.get('мітка_часу', ''), "Навантаження (%)": f"{consumption_percent:.1f}", "Частота (Гц)": f"{freq_dev:.2f}", "Оцінка (стара)": f"{initial_state:.2f}", "Вердикт (старий)": initial_verdict, "Оцінка (нова)": f"{optimized_state:.2f}", "Вердикт (новий)": optimized_verdict }
                self.report_data.append(report_row)
                if initial_verdict in stats_initial: stats_initial[initial_verdict] += 1
                if optimized_verdict in stats_optimized: stats_optimized[optimized_verdict] += 1
                sum_initial += float(initial_state); sum_optimized += float(optimized_state)
                if optimized_state < initial_state: improvements += 1
                elif optimized_state > initial_state: regressions += 1
                valid_rows += 1
            except Exception as e:
                print(f"Помилка при обробці рядка {index+2}: {e}. Рядок пропущено."); continue
            processed_attempts += 1
            if processed_attempts % update_step == 0 or valid_rows >= total_to_process:
                progress = processed_attempts / total_to_process if total_to_process else 1.0
                self.progressbar.set(progress); self.progress_label.configure(text=f"Обробка... {int(progress*100)}%"); self.update_idletasks()

        avg_initial = (sum_initial / valid_rows) if valid_rows else 0.0
        avg_optimized = (sum_optimized / valid_rows) if valid_rows else 0.0
        self.report_summary = {
            "rows": valid_rows,
            "initial_counts": stats_initial,
            "optimized_counts": stats_optimized,
            "avg_initial": avg_initial,
            "avg_optimized": avg_optimized,
            "improvements": improvements,
            "regressions": regressions
        }
        self.populate_report_table_optimized()
        self.button_load.configure(state="normal"); self.button_save.configure(state="normal"); self.progressbar.pack_forget(); self.progress_label.configure(text=f"Аналіз завершено. Оброблено рядків: {len(self.report_data)}")
        try:
            self.analysis_limit_slider.configure(state="normal")
        except Exception:
            pass

    def _update_limit_label(self, val):
        try:
            n = int(float(val))
        except Exception:
            n = 0
        self.analysis_limit_label.configure(text=f"Макс рядків: {n}")

    def populate_report_table_optimized(self):
        for widget in self.report_frame.winfo_children(): widget.destroy()
        self.create_table_header()
        display_limit = 100; data_to_display = self.report_data[:display_limit]
        if self.report_summary:
            sum_text = (
                f"ЗВЕДЕННЯ АНАЛІЗУ:\n"
                f"Оброблено рядків: {self.report_summary['rows']}\n"
                f"Середня оцінка (стара): {self.report_summary['avg_initial']:.2f}    (нова): {self.report_summary['avg_optimized']:.2f}\n"
                f"Кількість покращень (оптимізована < стара): {self.report_summary['improvements']}\n"
                f"Кількість регресій (оптимізована > стара): {self.report_summary['regressions']}\n\n"
                f"Розподіл вердиктів (старий): Stable={self.report_summary['initial_counts'].get('Стабільний',0)}, Warning={self.report_summary['initial_counts'].get('Попередження',0)}, Danger={self.report_summary['initial_counts'].get('Небезпека',0)}\n"
                f"Розподіл вердиктів (новий):  Stable={self.report_summary['optimized_counts'].get('Стабільний',0)}, Warning={self.report_summary['optimized_counts'].get('Попередження',0)}, Danger={self.report_summary['optimized_counts'].get('Небезпека',0)}\n"
            )
            summary_box = customtkinter.CTkTextbox(self.report_frame, height=140, font=("Arial", 12))
            summary_box.insert("1.0", sum_text)
            summary_box.configure(state="disabled")
            summary_box.pack(padx=10, pady=8, fill="x")
        info_text = f"Показано перші {len(data_to_display)} з {len(self.report_data)} рядків. Повний звіт буде збережено у файл."
        info_label = customtkinter.CTkLabel(self.report_frame, text=info_text, font=("Arial", 14)); info_label.pack(pady=6, fill="x")
        for report_row in data_to_display:
            row_frame = customtkinter.CTkFrame(self.report_frame); row_frame.pack(fill="x", expand=True, pady=2)
            _, initial_color = self._get_state_display(float(report_row['Оцінка (стара)']))
            _, optimized_color = self._get_state_display(float(report_row['Оцінка (нова)']))
            columns = list(report_row.keys())
            for i, col_name in enumerate(columns):
                value = report_row[col_name]
                text_color = "white"
                if col_name == "Вердикт (старий)": text_color = initial_color
                elif col_name == "Вердикт (новий)": text_color = optimized_color
                label = customtkinter.CTkLabel(row_frame, text=str(value), wraplength=150, text_color=text_color, font=("Arial", 12)); label.grid(row=0, column=i, padx=5, pady=2, sticky="ew")
            for col in range(len(columns)):
                row_frame.grid_columnconfigure(col, weight=1)
    def save_report(self):
        if not self.report_data:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")]);
        if not filepath: return
        report_df = pd.DataFrame(self.report_data)
        report_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        try:
            base, ext = os.path.splitext(filepath)
            summary_path = base + "_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                if self.report_summary:
                    f.write("ЗВЕДЕННЯ АНАЛІЗУ\n")
                    f.write(f"Оброблено рядків: {self.report_summary['rows']}\n")
                    f.write(f"Середня оцінка (стара): {self.report_summary['avg_initial']:.2f}\n")
                    f.write(f"Середня оцінка (нова): {self.report_summary['avg_optimized']:.2f}\n")
                    f.write(f"Покращень: {self.report_summary['improvements']}\n")
                    f.write(f"Регресій: {self.report_summary['regressions']}\n\n")
                    f.write("Розподіл вердиктів (старий):\n")
                    for k,v in self.report_summary['initial_counts'].items(): f.write(f"  {k}: {v}\n")
                    f.write("\nРозподіл вердиктів (новий):\n")
                    for k,v in self.report_summary['optimized_counts'].items(): f.write(f"  {k}: {v}\n")
                else:
                    f.write("Зведення відсутнє.\n")
        except Exception as e:
            print(f"Помилка при збереженні summary: {e}")

    def _get_state_display(self, state_value):
        if state_value <= 3.5: return "Стабільний", "#33FF57"
        elif state_value <= 7.0: return "Попередження", "#FFC300"
        else: return "Небезпека", "#FF5733"

    def show_analysis_summary(self):
        for widget in self.report_frame.winfo_children(): widget.destroy()
        if self.report_summary:
            header = customtkinter.CTkLabel(self.report_frame, text="Динамічний аналіз ефективності оптимізованої системи", font=("Arial", 16, "bold")); header.pack(pady=(10,5), anchor="w")
            text = (
                f"Оброблено рядків: {self.report_summary['rows']}\n"
                f"Середня оцінка (стара): {self.report_summary['avg_initial']:.2f}\n"
                f"Середня оцінка (нова): {self.report_summary['avg_optimized']:.2f}\n"
                f"Покращень: {self.report_summary['improvements']}; Регресій: {self.report_summary['regressions']}\n\n"
                "Висновок: оптимізована система показала загальне зміщення в напрямку зниження середнього ризику (якщо avg_optimized < avg_initial),\n"
                "а також конкретні покращення у випадках, позначених як 'покращення'. Рекомендується переглянути випадки регресії та\n"
                "підстави їх появи перед впровадженням у продуктивне середовище."
            )
            textbox = customtkinter.CTkTextbox(self.report_frame, height=260, font=("Arial", 12), wrap="word")
            textbox.insert("1.0", text); textbox.configure(state="disabled"); textbox.pack(padx=10, pady=10, fill="both", expand=True)

            dist_text = ("Розподіл вердиктів (старий):\n" + 
                         "\n".join([f"{k}: {v}" for k,v in self.report_summary['initial_counts'].items()]) +
                         "\n\nРозподіл вердиктів (новий):\n" +
                         "\n".join([f"{k}: {v}" for k,v in self.report_summary['optimized_counts'].items()]))
            dist_box = customtkinter.CTkTextbox(self.report_frame, height=140, font=("Arial", 12), wrap="word")
            dist_box.insert("1.0", dist_text); dist_box.configure(state="disabled"); dist_box.pack(padx=10, pady=(0,10), fill="both", expand=True)
        else:
            header = customtkinter.CTkLabel(self.report_frame, text="Аналіз ефективності оптимізованої системи", font=("Arial", 16, "bold")); header.pack(pady=(10, 5), anchor="w")
            textbox = customtkinter.CTkTextbox(self.report_frame, height=420, font=("Arial", 12), wrap="word")
            textbox.insert("1.0", (
                "Аналіз ефективності оптимізованої експертної системи:\n\n"
                "Звіт чітко демонструє значну ефективність оптимізованої експертної системи порівняно з початковою версією. "
                "Ключові аспекти, що підтверджують це:\n"
                "- Розширення бази правил: Збільшення кількості правил з 6 до 20 дозволило системі точніше реагувати на складні та нетипові ситуації, уникаючи хибних спрацьовувань.\n"
                "- Підвищена чутливість: Звуження терму 'Стабільна' для частоти дозволило системі раніше виявляти потенційні загрози, що є критично важливим для оперативного реагування.\n"
                "- Повнота аналізу та логічність поведінки: 3D-візуалізація поверхні відгуку підтверджує, що оптимізована система демонструє більш плавну та передбачувану поведінку.\n"
                "- Наближення до експертної логіки: Завдяки оптимізації та розширенню бази знань система наблизилась до логіки реального експерта-диспетчера.\n\n"
                "Рекомендації щодо нарощування вибірки:\n\n"
                "- Збільшити обсяг даних для кращої валідації.\n"
                "- Переглянути випадки регресії та уточнити правила для цих сценаріїв.\n"
            ))
            textbox.configure(state="disabled"); textbox.pack(padx=10, pady=10, fill="both", expand=True)

if __name__ == "__main__":
    app = App()
    app.mainloop()
