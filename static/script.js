(function() {
    const messages = document.getElementById("messages");
    const textarea = document.getElementById("prompt");
    const sendButton = document.getElementById("sendButton");
    const typing = document.getElementById("typing");
    const sidebar = document.querySelector(".sidebar");

    let generating = false;
    let totalTokens = 0;

    // --- Token counter element ---
    const tokenBadge = document.createElement("div");
    tokenBadge.id = "token-counter";
    tokenBadge.style.cssText =
        "position:fixed;bottom:12px;right:12px;background:#2d2d2d;color:#888;padding:4px 10px;border-radius:12px;font-size:12px;font-family:sans-serif;z-index:999;pointer-events:none;";
    tokenBadge.textContent = "Tokens: 0";
    document.body.appendChild(tokenBadge);

    function updateTokenCounter(text) {
        totalTokens += Math.max(0, Math.round(text.length / 4));
        tokenBadge.textContent = "Tokens: ~" + totalTokens;
    }

    // --- Typing indicator (3 bouncing dots) ---
    typing.innerHTML =
        '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';

    // Inject typing-dot CSS if not already present
    if (!document.getElementById("typing-dot-style")) {
        const style = document.createElement("style");
        style.id = "typing-dot-style";
        style.textContent = `
            #typing { display: none; align-items: center; gap: 4px; padding: 8px 0; }
            .typing-dot {
                display: inline-block;
                width: 8px; height: 8px;
                border-radius: 50%;
                background: #888;
                animation: typingBounce 1.4s infinite ease-in-out both;
            }
            .typing-dot:nth-child(1) { animation-delay: -0.32s; }
            .typing-dot:nth-child(2) { animation-delay: -0.16s; }
            .typing-dot:nth-child(3) { animation-delay: 0s; }
            @keyframes typingBounce {
                0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
                40% { transform: scale(1); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
    }

    // Auto-expand textarea
    textarea.addEventListener("input", () => {
        textarea.style.height = "auto";
        textarea.style.height = textarea.scrollHeight + "px";
    });

    // Enter sends, Shift+Enter newline
    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (!generating) sendMessage();
        }
    });

    sendButton.addEventListener("click", () => {
        if (!generating) sendMessage();
    });

    function scrollBottom() {
        messages.scrollTo({ top: messages.scrollHeight, behavior: "smooth" });
    }

    // --- Copy button on <pre><code> blocks ---
    function addCopyButtons() {
        document.querySelectorAll("pre code").forEach((block) => {
            const pre = block.parentElement;
            if (pre.querySelector(".copy-btn")) return;
            const btn = document.createElement("button");
            btn.className = "copy-btn";
            btn.textContent = "\uD83D\uDCCB Copiar";
            Object.assign(btn.style, {
                position: "absolute",
                top: "6px",
                right: "8px",
                background: "rgba(255,255,255,0.1)",
                border: "none",
                color: "#ccc",
                padding: "2px 10px",
                borderRadius: "4px",
                cursor: "pointer",
                fontSize: "12px",
                fontFamily: "sans-serif",
                zIndex: "2",
                transition: "background 0.2s",
            });
            btn.addEventListener("mouseenter", () => (btn.style.background = "rgba(255,255,255,0.2)"));
            btn.addEventListener("mouseleave", () => (btn.style.background = "rgba(255,255,255,0.1)"));
            btn.addEventListener("click", async () => {
                const text = block.textContent;
                try {
                    await navigator.clipboard.writeText(text);
                } catch {
                    const ta = document.createElement("textarea");
                    ta.value = text;
                    document.body.appendChild(ta);
                    ta.select();
                    document.execCommand("copy");
                    document.body.removeChild(ta);
                }
                btn.textContent = "\u2705 Copiado!";
                setTimeout(() => (btn.textContent = "\uD83D\uDCCB Copiar"), 2000);
            });
            pre.style.position = "relative";
            pre.appendChild(btn);
        });
    }

    // --- Feedback buttons ---
    function addFeedback(bubble, fullText) {
        const container = document.createElement("div");
        container.className = "feedback-row";
        Object.assign(container.style, {
            display: "flex",
            gap: "6px",
            justifyContent: "flex-end",
            marginTop: "6px",
        });

        [["\uD83D\uDC4D", "positive"], ["\uD83D\uDC4E", "negative"]].forEach(([emoji, type]) => {
            const btn = document.createElement("button");
            btn.textContent = emoji;
            Object.assign(btn.style, {
                background: "none",
                border: "1px solid transparent",
                borderRadius: "4px",
                cursor: "pointer",
                fontSize: "16px",
                padding: "0 4px",
                opacity: "0.5",
                transition: "opacity 0.2s",
            });
            btn.addEventListener("mouseenter", () => (btn.style.opacity = "1"));
            btn.addEventListener("mouseleave", () => (btn.style.opacity = "0.5"));
            btn.addEventListener("click", async () => {
                try {
                    await fetch("/api/feedback", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ feedback: type, message: fullText }),
                    });
                } catch (e) {
                    console.error("Feedback error:", e);
                }
                btn.style.opacity = "1";
                btn.style.borderColor = "#888";
                container.querySelectorAll("button").forEach((b) => (b.style.pointerEvents = "none"));
            });
            container.appendChild(btn);
        });

        bubble.appendChild(container);
    }

    function createMessage(content, type) {
        const container = document.createElement("div");
        container.className = "message " + type;

        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.innerText = type === "user" ? "H" : "AI";

        const bubble = document.createElement("div");
        bubble.className = "bubble";

        container.appendChild(avatar);
        container.appendChild(bubble);
        messages.appendChild(container);

        if (type === "ai" && content) {
            bubble.innerHTML = content;
            addCopyButtons();
            addFeedback(bubble, content);
            updateTokenCounter(content);
        } else if (type === "ai") {
            bubble.innerHTML = content;
        } else {
            bubble.innerHTML = content;
            updateTokenCounter(content);
        }

        scrollBottom();
        return bubble;
    }

    async function sendMessage() {
        const prompt = textarea.value.trim();
        if (!prompt) return;

        generating = true;
        createMessage(prompt, "user");

        textarea.value = "";
        textarea.style.height = "auto";

        typing.style.display = "flex";
        scrollBottom();

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: prompt }),
            });

            typing.style.display = "none";

            if (!response.ok) {
                const err = await response.json().catch(() => ({ erro: "Erro desconhecido" }));
                createMessage("⚠️ " + (err.erro || "Erro no servidor."), "ai");
                generating = false;
                return;
            }

            const bubble = createMessage("", "ai");
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullText = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                fullText += chunk;

                bubble.innerHTML = marked.parse(fullText);

                document.querySelectorAll("pre code").forEach((block) => {
                    hljs.highlightElement(block);
                });

                addCopyButtons();
                scrollBottom();
            }

            addFeedback(bubble, fullText);
            updateTokenCounter(fullText);
        } catch (err) {
            typing.style.display = "none";
            createMessage("❌ Erro ao se comunicar com o servidor: " + err.message, "ai");
            console.error(err);
        }

        generating = false;
    }

    // --- Theme toggle ---
    const themeBtn = document.createElement("button");
    themeBtn.id = "theme-toggle";
    themeBtn.innerHTML = "\uD83C\uDF19";
    Object.assign(themeBtn.style, {
        background: "none",
        border: "1px solid #444",
        borderRadius: "6px",
        color: "#ccc",
        cursor: "pointer",
        fontSize: "18px",
        padding: "6px 10px",
        marginTop: "auto",
    });
    themeBtn.title = "Alternar tema";

    if (sidebar) {
        sidebar.appendChild(themeBtn);
    } else {
        document.body.appendChild(themeBtn);
        Object.assign(themeBtn.style, {
            position: "fixed",
            bottom: "12px",
            left: "12px",
            zIndex: "999",
        });
    }

    function applyTheme(dark) {
        const root = document.documentElement;
        if (dark) {
            root.removeAttribute("data-theme");
            themeBtn.innerHTML = "\uD83C\uDF19";
        } else {
            root.setAttribute("data-theme", "light");
            themeBtn.innerHTML = "\uD83C\uDF1E";
        }
        localStorage.setItem("theme", dark ? "dark" : "light");
    }

    const saved = localStorage.getItem("theme");
    if (saved === "light") {
        applyTheme(false);
    }

    themeBtn.addEventListener("click", () => {
        const isDark = !document.documentElement.hasAttribute("data-theme");
        applyTheme(!isDark);
    });
})();
