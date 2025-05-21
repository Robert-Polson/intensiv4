import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import joblib

model_package = joblib.load('model.pth')
models = model_package['models']
vectorizer = model_package['vectorizer']
categories = model_package['categories']

def predict_category():
    user_input = comment_text.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showwarning("Предупреждение", "Пожалуйста, введите комментарий.")
        return
    comment_tok = ' '.join(user_input.lower().split())
    comment_vec = vectorizer.transform([comment_tok])
    predicted_probs = {}
    for category in categories:
        model = models.get(category)
        if model:
            proba = model.predict_proba(comment_vec)[0][1]
            predicted_probs[category] = proba
    if predicted_probs:
        predicted_category = max(predicted_probs, key=predicted_probs.get)
        result_text = f"Комментарий:\n{user_input}\n\n" \
                      f"Категория: {predicted_category}"
        result_text_widget.config(state='normal')
        result_text_widget.delete("1.0", tk.END)
        result_text_widget.insert(tk.END, result_text)
        result_text_widget.config(state='disabled')

root = tk.Tk()
root.title("Классификатор коментариев")
root.geometry("800x600")
instruction_label = tk.Label(root,fg="#000000", text="Введите комментарий и нажмите 'Предсказать'", font=("Helvetica Neue", 18))
instruction_label.pack(pady=10)
input_frame = tk.Frame(root, bd=2, relief=tk.RIDGE)
input_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
input_label = tk.Label(input_frame, text="Ваш комментарий:", font=("Helvetica Neue", 16), fg="#000000")
input_label.pack(anchor='nw', padx=10, pady=5)
comment_text = tk.Text(input_frame, height=8, font=("Helvetica Neue", 16), wrap=tk.WORD, bd=0, highlightthickness=0)
comment_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)


style = ttk.Style()
style.theme_use('clam')
style.configure('TButton',
                font=('Helvetica Neue', 18, 'bold'),
                
                foreground='#222831',
                padding=12)
style.map('TButton',
          background=[('active', '#7af760')],
          foreground=[('active', '#222831')])
predict_button = ttk.Button(root, text="Предсказать", command=predict_category)
predict_button.pack(pady=20)
result_container = tk.Frame(root)
result_container.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
result_text_widget = tk.Text(result_container, height=10, font=("Helvetica Neue", 16), fg="#000000", wrap=tk.WORD, state='disabled')
result_text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)


root.mainloop()