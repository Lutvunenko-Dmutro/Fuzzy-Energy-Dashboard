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

    def update_interactive_values(self, _=None): # _=None приймає значення від повзунка, але ми його ігноруємо
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
        self.progress_label = customtkinter.CTkLabel(top_frame, text=""); self.progress_label.pack(side="left", padx=10, pady=10)
        self.progressbar = customtkinter.CTkProgressBar(top_frame, orientation="horizontal"); self.progressbar.set(0)
        self.report_frame = customtkinter.CTkScrollableFrame(tab, label_text="Звіт по аналізу даних"); self.report_frame.pack(padx=10, pady=10, fill="both", expand=True)
        self.create_table_header()
    def create_rules_tab(self, tab):
        rules_frame = customtkinter.CTkScrollableFrame(tab); rules_frame.pack(padx=20, pady=20, fill="both", expand=True)
        initial_label = customtkinter.CTkLabel(rules_frame, text="База правил початкової системи (6 правил)", font=("Arial", 18, "bold")); initial_label.pack(pady=(10, 5), anchor="w")
        initial_rules_text = ("1. ЯКЩО споживання 'Normal' І частота 'Stable' ТО стан 'Stable'\n2. ЯКЩО споживання 'High' І частота 'Negative_Low' ТО стан 'Warning'\n3. ЯКЩО споживання 'Critical' І частота 'Negative_High' ТО стан 'Danger'\n4. ЯКЩО споживання 'Low' І частота 'Positive' ТО стан 'Warning'\n5. ЯКЩО споживання 'High' АБО 'Critical' ТО стан 'Danger'\n6. ЯКЩО споживання 'Low' ТО стан 'Stable'")
        initial_textbox = customtkinter.CTkTextbox(rules_frame, height=150, font=("Arial", 14)); initial_textbox.insert("1.0", initial_rules_text); initial_textbox.configure(state="disabled"); initial_textbox.pack(pady=10, fill="x", expand=True)
        optimized_label = customtkinter.CTkLabel(rules_frame, text="База правил оптимізованої системи (20 правил)", font=("Arial", 18, "bold")); optimized_label.pack(pady=(20, 5), anchor="w")
        optimized_rules_text = ("--- Стабільні сценарії ---\n1. ЯКЩО споживання 'Low' І частота 'Stable' ТО стан 'Stable'\n2. ЯКЩО споживання 'Low' І частота 'Negative_Low' ТО стан 'Stable'\n3. ЯКЩО споживання 'Low' І частота 'Positive' ТО стан 'Stable'\n4. ЯКЩО споживання 'Normal' І частота 'Stable' ТО стан 'Stable'\n--- Сценарії попередження ---\n5. ЯКЩО споживання 'Low' І частота 'Negative_High' ТО стан 'Warning'\n6. ЯКЩО споживання 'Normal' І частота 'Negative_Low' ТО стан 'Warning'\n7. ЯКЩО споживання 'Normal' І частота 'Positive' ТО стан 'Warning'\n8. ЯКЩО споживання 'High' І частота 'Stable' ТО стан 'Warning'\n9. ЯКЩО споживання 'High' І частота 'Negative_Low' ТО стан 'Warning'\n10. ЯКЩО споживання 'High' І частота 'Positive' ТО стан 'Warning'\n11. ЯКЩО споживання 'Critical' І частота 'Stable' ТО стан 'Warning'\n--- Небезпечні сценарії ---\n12. ЯКЩО споживання 'Critical' І частота 'Positive' ТО стан 'Danger'\n13. ЯКЩО споживання 'Normal' І частота 'Negative_High' ТО стан 'Danger'\n14. ЯКЩО споживання 'High' І частота 'Negative_High' ТО стан 'Danger'\n15. ЯКЩО споживання 'Critical' І частота 'Negative_Low' ТО стан 'Danger'\n16. ЯКЩО споживання 'Critical' І частота 'Negative_High' ТО стан 'Danger'\n--- Загальні правила безпеки ---\n17. ЯКЩО споживання 'Critical' ТО стан 'Danger'\n18. ЯКЩО частота 'Negative_High' ТО стан 'Danger'\n19. ЯКЩО споживання 'High' ТО стан 'Warning'\n20. ЯКЩО споживання 'Low' ТО стан 'Stable'")
        optimized_textbox = customtkinter.CTkTextbox(rules_frame, height=450, font=("Arial", 14)); optimized_textbox.insert("1.0", optimized_rules_text); optimized_textbox.configure(state="disabled"); optimized_textbox.pack(pady=10, fill="x", expand=True)
    def create_viz_tab(self, tab):
        top_frame = customtkinter.CTkFrame(tab); top_frame.pack(padx=10, pady=10, fill="x")
        self.viz_button = customtkinter.CTkButton(top_frame, text="Побудувати поверхні відгуку", command=self.generate_and_plot_surfaces); self.viz_button.pack(side="left", padx=10, pady=10)
        self.viz_label = customtkinter.CTkLabel(top_frame, text="Натисніть кнопку, щоб почати розрахунок (може зайняти кілька секунд)"); self.viz_label.pack(side="left", padx=10, pady=10)
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
        self.viz_label.configure(text="Йде розрахунок..."); self.viz_button.configure(state="disabled"); self.update_idletasks()
        cons_range = np.linspace(0, 100, 30); freq_range = np.linspace(-0.5, 0.5, 30); x, y = np.meshgrid(cons_range, freq_range); z_initial = np.zeros_like(x); z_optimized = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]): z_initial[i, j] = get_initial_system_state(x[i, j], y[i, j]); z_optimized[i, j] = get_optimized_system_state(x[i, j], y[i, j])
        self.ax1_3d.clear(); self.ax2_3d.clear()
        self.ax1_3d.plot_surface(x, y, z_initial, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True); self.ax1_3d.set_title('Початкова система'); self.ax1_3d.set_xlabel('Споживання (%)'); self.ax1_3d.set_ylabel('Відхилення частоти (Гц)'); self.ax1_3d.set_zlabel('Рівень загрози')
        self.ax2_3d.plot_surface(x, y, z_optimized, rstride=1, cstride=1, cmap='plasma', linewidth=0.4, antialiased=True); self.ax2_3d.set_title('Оптимізована система'); self.ax2_3d.set_xlabel('Споживання (%)'); self.ax2_3d.set_ylabel('Відхилення частоти (Гц)'); self.ax2_3d.set_zlabel('Рівень загрози')
        self.canvas_3d.draw(); self.viz_label.configure(text="Розрахунок завершено."); self.viz_button.configure(state="normal")
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
        header_frame.grid_columnconfigure(tuple(range(len(headers))), weight=1)
    def process_csv_file(self):
        filepath = 'power_load_hourly.csv'; self.create_table_header(); self.report_data.clear()
        self.button_load.configure(state="disabled"); self.button_save.configure(state="disabled"); self.progress_label.configure(text="Обробка..."); self.progressbar.pack(side="left", padx=10, pady=10, fill="x", expand=True); self.update_idletasks()
        try:
            df = pd.read_csv(filepath); df.columns = df.columns.str.lower()
        except FileNotFoundError:
            self.create_table_header(); error_label = customtkinter.CTkLabel(self.report_frame, text=f"ПОМИЛКА: Файл '{filepath}' не знайдено!", text_color="red", font=("Arial", 14)); error_label.pack(pady=20)
            self.button_load.configure(state="normal"); self.progressbar.pack_forget(); self.progress_label.configure(text="")
            return
        total_rows = len(df)
        for index, row in df.iterrows():
            try:
                if row['потужність_мвт'] == 0: continue
                consumption_percent = (row['навантаження_мвт'] / row['потужність_мвт']) * 100
                freq_dev = 0.0; load_ratio = consumption_percent / 100
                if load_ratio > 0.75: freq_dev = -0.1 * (load_ratio - 0.75) * 4
                elif load_ratio < 0.30: freq_dev = 0.05 * (0.30 - load_ratio) * 3.33
                initial_state = get_initial_system_state(consumption_percent, freq_dev); optimized_state = get_optimized_system_state(consumption_percent, freq_dev)
                initial_verdict, _ = self._get_state_display(initial_state); optimized_verdict, _ = self._get_state_display(optimized_state)
                report_row = { "Час": row['мітка_часу'], "Навантаження (%)": f"{consumption_percent:.1f}", "Частота (Гц)": f"{freq_dev:.2f}", "Оцінка (стара)": f"{initial_state:.2f}", "Вердикт (старий)": initial_verdict, "Оцінка (нова)": f"{optimized_state:.2f}", "Вердикт (новий)": optimized_verdict }
                self.report_data.append(report_row)
            except Exception as e:
                print(f"Помилка при обробці рядка {index+2}: {e}. Рядок пропущено."); continue
            if (index + 1) % 100 == 0:
                progress = (index + 1) / total_rows; self.progressbar.set(progress); self.progress_label.configure(text=f"Обробка... {int(progress*100)}%"); self.update_idletasks()
        self.populate_report_table_optimized()
        self.button_load.configure(state="normal"); self.button_save.configure(state="normal"); self.progressbar.pack_forget(); self.progress_label.configure(text=f"Аналіз завершено. Оброблено рядків: {len(self.report_data)}")
    def populate_report_table_optimized(self):
        for widget in self.report_frame.winfo_children(): widget.destroy()
        self.create_table_header()
        display_limit = 100; data_to_display = self.report_data[:display_limit]
        info_text = f"Показано перші {len(data_to_display)} з {len(self.report_data)} рядків. Повний звіт буде збережено у файл."
        info_label = customtkinter.CTkLabel(self.report_frame, text=info_text, font=("Arial", 14)); info_label.pack(pady=10, fill="x")
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
            row_frame.grid_columnconfigure(tuple(range(len(columns))), weight=1)
    def save_report(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")]);
        if not filepath: return
        report_df = pd.DataFrame(self.report_data); report_df.to_csv(filepath, index=False, encoding='utf-8-sig')
    def _get_state_display(self, state_value):
        if state_value <= 3.5: return "Стабільний", "#33FF57"
        elif state_value <= 7.0: return "Попередження", "#FFC300"
        else: return "Небезпека", "#FF5733"
        
if __name__ == "__main__":
    app = App()
    app.mainloop()