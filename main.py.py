import warnings
import pandas as pd
from datasets import load_dataset
from transformers import pipeline
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog, ttk
from datetime import datetime
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# Load model and dataset
dataset = load_dataset("dair-ai/emotion")
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Rich dark theme colors
DARK_BG = "#121212"
PANEL_BG = "#1f1f2e"
TEXT_COLOR = "#e0e0e0"
LABEL_COLOR = "#f5f5f5"
BUTTON_PRIMARY = "#4cafff"
BUTTON_SECONDARY = "#ff9800"
HERO_TEXT = "#ffffff"
HERO_SUBTEXT = "#a0a0a0"

# Emotion colors and emojis
emotion_styles = {
    "love": {"color": "#ff4c4c", "emoji": "❤️"},
    "joy": {"color": "#ffb74c", "emoji": "😊"},
    "sadness": {"color": "#4c6ef5", "emoji": "😢"},
    "anger": {"color": "#d32f2f", "emoji": "😡"},
    "fear": {"color": "#9c27b0", "emoji": "😨"},
    "surprise": {"color": "#00c853", "emoji": "😲"},
    "neutral": {"color": "#888888", "emoji": "😐"}  # Added neutral
}

# Adjust emotion for "love"
def adjust_emotion(text, result):
    lowered = text.lower()
    if result['label'] == "joy":
        if any(word in lowered for word in [
            "love", "lover", "beloved", "cherish", "affection",
            "in love", "falling for", "my heart", "adore", "fond of"
        ]):
            if "don't love" not in lowered and "not love" not in lowered:
                result['label'] = "love"
    # Map unknown labels to neutral
    if result['label'] not in emotion_styles:
        result['label'] = "neutral"
    return result

# Global results
results = []

# Analyze text
def analyze_text():
    global results
    results = []
    output_area.configure(state='normal')
    output_area.delete(1.0, tk.END)

    text = input_area.get(1.0, tk.END).strip()
    if not text:
        messagebox.showwarning("Warning", "Please enter some text!")
        return

    sentences = [s.strip() for s in text.split('.') if s.strip()]
    for sentence in sentences:
        result = emotion_pipeline(sentence)[0]
        result = adjust_emotion(sentence, result)
        detected_emotion = result['label']

        output_area.insert(tk.END, f"Text: {sentence}\nDetected Emotion: ")
        style = emotion_styles.get(detected_emotion, {"color": "#ffffff", "emoji": ""})
        output_area.insert(tk.END, detected_emotion, detected_emotion)
        output_area.insert(tk.END, f" {style['emoji']}\n\n")

        results.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": sentence,
            "emotion": detected_emotion
        })

    output_area.configure(state='disabled')

# Show emotion timeline chart
def show_emotion_timeline():
    if not results:
        messagebox.showwarning("Warning", "No results to visualize!")
        return

    # Safe mapping for all emotions in results
    all_emotions = list(emotion_styles.keys())
    emotion_to_int = {e: i for i, e in enumerate(all_emotions)}

    x = list(range(1, len(results)+1))
    y = []
    colors = []

    for r in results:
        emotion = r["emotion"]
        if emotion not in emotion_to_int:
            emotion = "neutral"  # fallback
        y.append(emotion_to_int[emotion])
        colors.append(emotion_styles.get(emotion, {"color": "#888888"})["color"])

    plt.style.use('dark_background')
    plt.figure(figsize=(9,5))
    plt.scatter(x, y, c=colors, s=120)
    plt.plot(x, y, color="#888888", alpha=0.5, linestyle="--")
    plt.yticks(list(emotion_to_int.values()), list(emotion_to_int.keys()))
    plt.xlabel("Sentence Number", color=TEXT_COLOR)
    plt.ylabel("Emotion", color=TEXT_COLOR)
    plt.title("Emotion Timeline Across Journal", color=HERO_TEXT)
    plt.grid(True, linestyle="--", alpha=0.5, color="#444444")
    plt.show()

# Save results
def save_results():
    if not results:
        messagebox.showwarning("Warning", "No results to save!")
        return
    filename = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV files", "*.csv")])
    if filename:
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        messagebox.showinfo("Success", f"Results saved to {filename}")

# Load text file
def load_file():
    filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if filename:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            input_area.delete(1.0, tk.END)
            input_area.insert(tk.END, content)

# Open saved journal CSV
def open_journal():
    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not filename:
        return
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open file: {e}")
        return

    journal_window = tk.Toplevel()
    journal_window.title(f"Journal Viewer - {filename}")
    journal_window.configure(bg=DARK_BG)
    journal_window.state('zoomed')  # Maximized

    cols = list(df.columns)
    tree = ttk.Treeview(journal_window, columns=cols, show="headings")
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview", background=PANEL_BG, foreground=TEXT_COLOR, fieldbackground=PANEL_BG)
    style.configure("Treeview.Heading", background=DARK_BG, foreground=LABEL_COLOR)

    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, width=150)

    for _, row in df.iterrows():
        tree.insert("", tk.END, values=list(row))

    tree.pack(fill=tk.BOTH, expand=True)

# Launch journal GUI
def launch_journal():
    global input_area, output_area

    journal_root = tk.Toplevel()
    journal_root.title("Daily Journal Emotion Detection")
    journal_root.configure(bg=DARK_BG)
    journal_root.state('zoomed')  # Maximized

    top_frame = tk.Frame(journal_root, bg=DARK_BG)
    top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    tk.Button(top_frame, text="Analyze", command=analyze_text, width=15, bg=BUTTON_PRIMARY, fg="#ffffff").pack(side=tk.LEFT, padx=5)
    tk.Button(top_frame, text="Load File", command=load_file, width=15, bg=BUTTON_PRIMARY, fg="#ffffff").pack(side=tk.LEFT, padx=5)
    tk.Button(top_frame, text="Save Results", command=save_results, width=15, bg=BUTTON_SECONDARY, fg="#ffffff").pack(side=tk.LEFT, padx=5)
    tk.Button(top_frame, text="Show Timeline", command=show_emotion_timeline, width=15, bg=BUTTON_SECONDARY, fg="#ffffff").pack(side=tk.LEFT, padx=5)

    main_frame = tk.Frame(journal_root, bg=DARK_BG)
    main_frame.pack(fill=tk.BOTH, expand=True)

    input_panel = tk.Frame(main_frame, bg=PANEL_BG)
    input_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    tk.Label(input_panel, text="Enter your journal text:", fg=TEXT_COLOR, bg=PANEL_BG).pack()
    input_area = scrolledtext.ScrolledText(input_panel, width=50, height=30, bg="#2b2b3b", fg=TEXT_COLOR, insertbackground=TEXT_COLOR)
    input_area.pack(fill=tk.BOTH, expand=True)

    output_panel = tk.Frame(main_frame, bg=PANEL_BG)
    output_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    tk.Label(output_panel, text="Detected Emotions:", fg=TEXT_COLOR, bg=PANEL_BG).pack()
    output_area = scrolledtext.ScrolledText(output_panel, width=50, height=30, state='disabled', bg="#2b2b3b", fg=TEXT_COLOR)
    output_area.pack(fill=tk.BOTH, expand=True)

    for emotion, style in emotion_styles.items():
        output_area.tag_config(emotion, foreground=style['color'], font=('Arial', 10, 'bold'))

# Main menu (hero page)
def main_menu():
    root = tk.Tk()
    root.title("Journal Manager")
    root.configure(bg=DARK_BG)
    root.state('zoomed')  # Maximized

    center_frame = tk.Frame(root, bg=DARK_BG)
    center_frame.place(relx=0.5, rely=0.5, anchor="center")

    tk.Label(center_frame, text="Welcome to Daily Journal", font=("Arial", 32, "bold"),
             fg=HERO_TEXT, bg=DARK_BG).pack(pady=20)
    tk.Label(center_frame, text="Track your emotions • Reflect • Grow", font=("Arial", 14),
             fg=HERO_SUBTEXT, bg=DARK_BG).pack(pady=5)

    tk.Button(center_frame, text="✍ Create New Journal", command=launch_journal,
              width=25, height=2, font=("Arial", 12, "bold"), bg=BUTTON_PRIMARY, fg="#ffffff").pack(pady=10)
    tk.Button(center_frame, text="📂 Open Existing Journal", command=open_journal,
              width=25, height=2, font=("Arial", 12, "bold"), bg=BUTTON_SECONDARY, fg="#ffffff").pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main_menu()
