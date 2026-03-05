/**
 * Fusor AI embeddable chat widget.
 * Load with: <script src="https://YOUR_API_URL/static/widget.js" data-api-url="..." data-user-id="..." data-bot-id="..." data-placement="bottom-corner"></script>
 */
(function () {
  "use strict";

  var PREFIX = "fusor-widget-";
  var script = document.currentScript;
  if (!script) return;

  var apiUrl = (script.getAttribute("data-api-url") || "").trim();
  if (!apiUrl) {
    try {
      var src = script.src;
      if (src) {
        var a = document.createElement("a");
        a.href = src;
        apiUrl = a.origin;
      }
    } catch (e) {}
  }
  if (!apiUrl) return;

  var userId = (script.getAttribute("data-user-id") || "").trim();
  var botId = (script.getAttribute("data-bot-id") || "").trim();
  if (!userId || !botId) return;

  var placement = (script.getAttribute("data-placement") || "bottom-corner").toLowerCase();
  var targetSelector = (script.getAttribute("data-target") || "").trim();

  var config = {
    chatbotName: "Chat",
    welcomeMessage: "Hello! How can I help you today?",
    color: "#2563eb",
    logo: null
  };

  function injectStyles() {
    if (document.getElementById(PREFIX + "styles")) return;
    var style = document.createElement("style");
    style.id = PREFIX + "styles";
    style.textContent = [
      "." + PREFIX + "root{ font-family: system-ui, -apple-system, sans-serif; font-size: 14px; box-sizing: border-box; }",
      "." + PREFIX + "root *{ box-sizing: border-box; }",
      "." + PREFIX + "button{ position: fixed; bottom: 20px; right: 20px; width: 56px; height: 56px; border-radius: 50%; border: none; cursor: pointer; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15); z-index: 2147483646; transition: transform 0.2s; }",
      "." + PREFIX + "button:hover{ transform: scale(1.05); }",
      "." + PREFIX + "button svg{ width: 28px; height: 28px; fill: white; }",
      "." + PREFIX + "panel{ position: fixed; bottom: 86px; right: 20px; width: 380px; max-width: calc(100vw - 40px); height: 500px; max-height: 80vh; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.12); display: flex; flex-direction: column; background: #fff; z-index: 2147483645; overflow: hidden; }",
      "." + PREFIX + "panel-inline{ position: relative; bottom: auto; right: auto; width: 100%; height: 400px; max-height: 70vh; }",
      "." + PREFIX + "header{ padding: 14px 16px; color: #fff; display: flex; align-items: center; gap: 10px; flex-shrink: 0; }",
      "." + PREFIX + "header img{ width: 32px; height: 32px; border-radius: 6px; object-fit: cover; }",
      "." + PREFIX + "header span{ font-weight: 600; font-size: 15px; }",
      "." + PREFIX + "messages{ flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 10px; background: #f8fafc; }",
      "." + PREFIX + "msg{ max-width: 85%; padding: 10px 12px; border-radius: 12px; line-height: 1.4; word-break: break-word; }",
      "." + PREFIX + "msg-user{ align-self: flex-end; background: #2563eb; color: #fff; }",
      "." + PREFIX + "msg-bot{ align-self: flex-start; background: #fff; border: 1px solid #e2e8f0; }",
      "." + PREFIX + "msg-error{ align-self: flex-start; background: #fef2f2; border: 1px solid #fecaca; color: #b91c1c; }",
      "." + PREFIX + "input-row{ padding: 12px; flex-shrink: 0; display: flex; gap: 8px; background: #fff; border-top: 1px solid #e2e8f0; }",
      "." + PREFIX + "input-row input{ flex: 1; padding: 10px 12px; border: 1px solid #e2e8f0; border-radius: 8px; font-size: 14px; outline: none; }",
      "." + PREFIX + "input-row input:focus{ border-color: #2563eb; }",
      "." + PREFIX + "input-row button{ padding: 10px 16px; border: none; border-radius: 8px; background: #2563eb; color: #fff; font-weight: 500; cursor: pointer; }",
      "." + PREFIX + "input-row button:disabled{ opacity: 0.6; cursor: not-allowed; }",
      "." + PREFIX + "hidden{ display: none !important; }"
    ].join("\n");
    document.head.appendChild(style);
  }

  function addMessage(container, text, role, isError) {
    var div = document.createElement("div");
    div.className = PREFIX + "msg " + (role === "user" ? PREFIX + "msg-user" : isError ? PREFIX + "msg-error" : PREFIX + "msg-bot");
    div.textContent = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
  }

  function wsBase() {
    var u = apiUrl;
    return (u.indexOf("https:") === 0 ? "wss:" : "ws:") + u.slice(u.indexOf(":"));
  }

  function sendMessage(text, messagesEl, inputEl, sendBtn) {
    if (!text.trim()) return;
    addMessage(messagesEl, text.trim(), "user");
    inputEl.value = "";
    sendBtn.disabled = true;

    var wsUrl = wsBase() + "/ws/chat/" + encodeURIComponent(userId) + "/" + encodeURIComponent(botId);
    var useRest = false;

    function showAnswer(answer, isError) {
      addMessage(messagesEl, answer || "Sorry, something went wrong.", "bot", isError);
      sendBtn.disabled = false;
    }

    try {
      var ws = new WebSocket(wsUrl);
      var resolved = false;

      ws.onopen = function () {
        ws.send(JSON.stringify({ message: text.trim() }));
      };

      ws.onmessage = function (ev) {
        if (resolved) return;
        try {
          var data = JSON.parse(ev.data);
          if (data.error) {
            showAnswer(data.error, true);
          } else if (data.answer != null) {
            showAnswer(data.answer, false);
          }
          resolved = true;
          ws.close();
        } catch (e) {
          showAnswer("Invalid response.", true);
          resolved = true;
        }
      };

      ws.onerror = function () {
        if (!resolved) {
          resolved = true;
          useRest = true;
        }
      };

      ws.onclose = function () {
        if (!resolved) {
          resolved = true;
          useRest = true;
        }
        if (useRest) restFallback();
      };

      setTimeout(function () {
        if (!resolved) {
          resolved = true;
          useRest = true;
          try { ws.close(); } catch (e) {}
          restFallback();
        }
      }, 25000);
    } catch (e) {
      restFallback();
    }

    function restFallback() {
      var xhr = new XMLHttpRequest();
      xhr.open("POST", apiUrl + "/query");
      xhr.setRequestHeader("Content-Type", "application/json");
      xhr.onreadystatechange = function () {
        if (xhr.readyState !== 4) return;
        sendBtn.disabled = false;
        try {
          if (xhr.status >= 200 && xhr.status < 300) {
            var res = JSON.parse(xhr.responseText);
            showAnswer(res.answer != null ? res.answer : "No response.", false);
          } else {
            showAnswer("Request failed. Please try again.", true);
          }
        } catch (e) {
          showAnswer("Request failed. Please try again.", true);
        }
      };
      xhr.onerror = function () {
        sendBtn.disabled = false;
        showAnswer("Network error. Please try again.", true);
      };
      try {
        xhr.send(JSON.stringify({ query: text.trim(), user_id: userId, bot_id: botId }));
      } catch (e) {
        sendBtn.disabled = false;
        showAnswer("Request failed.", true);
      }
    }
  }

  function buildPanel(messagesEl, inputEl, sendBtn) {
    var header = document.createElement("div");
    header.className = PREFIX + "header";
    header.style.backgroundColor = config.color;
    if (config.logo) {
      var img = document.createElement("img");
      img.src = config.logo;
      img.alt = config.chatbotName;
      header.appendChild(img);
    }
    var title = document.createElement("span");
    title.textContent = config.chatbotName;
    header.appendChild(title);

    var welcome = document.createElement("div");
    welcome.className = PREFIX + "msg " + PREFIX + "msg-bot";
    welcome.textContent = config.welcomeMessage;
    messagesEl.appendChild(welcome);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    var inputRow = document.createElement("div");
    inputRow.className = PREFIX + "input-row";
    var input = document.createElement("input");
    input.type = "text";
    input.placeholder = "Type your message...";
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") sendMessage(input.value, messagesEl, input, sendBtn);
    });
    var btn = document.createElement("button");
    btn.textContent = "Send";
    btn.addEventListener("click", function () {
      sendMessage(input.value, messagesEl, input, btn);
    });
    inputRow.appendChild(input);
    inputRow.appendChild(btn);

    return { header: header, inputRow: inputRow, input: input, sendBtn: btn };
  }

  function render() {
    injectStyles();

    var root = document.createElement("div");
    root.className = PREFIX + "root";

    var panel = document.createElement("div");
    panel.className = PREFIX + "panel" + (placement === "inline" ? " " + PREFIX + "panel-inline" : "");
    panel.id = PREFIX + "panel";

    var messagesEl = document.createElement("div");
    messagesEl.className = PREFIX + "messages";

    var inputEl = document.createElement("input");
    var sendBtn = document.createElement("button");
    var parts = buildPanel(messagesEl, inputEl, sendBtn);

    panel.appendChild(parts.header);
    panel.appendChild(messagesEl);
    panel.appendChild(parts.inputRow);
    root.appendChild(panel);

    if (placement !== "inline") {
      var btn = document.createElement("button");
      btn.className = PREFIX + "button";
      btn.style.backgroundColor = config.color;
      btn.title = config.chatbotName;
      btn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H5.17L4 17.17V4h16v12z"/></svg>';
      btn.addEventListener("click", function () {
        var p = document.getElementById(PREFIX + "panel");
        if (p.classList.contains(PREFIX + "hidden")) {
          p.classList.remove(PREFIX + "hidden");
        } else {
          p.classList.add(PREFIX + "hidden");
        }
      });
      root.appendChild(btn);
      panel.classList.add(PREFIX + "hidden");
    }

    var container = placement === "inline" && targetSelector
      ? document.querySelector(targetSelector)
      : document.body;
    if (!container) container = document.body;
    container.appendChild(root);

    window._fusorWidgetSend = function (text) {
      sendMessage(text, messagesEl, parts.input, parts.sendBtn);
    };
  }

  function loadConfig(cb) {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", apiUrl + "/chatbot-config/" + encodeURIComponent(userId) + "/" + encodeURIComponent(botId));
    xhr.onreadystatechange = function () {
      if (xhr.readyState !== 4) {
        cb();
        return;
      }
      try {
        if (xhr.status >= 200 && xhr.status < 300) {
          var res = JSON.parse(xhr.responseText);
          if (res.chatbot_name) config.chatbotName = res.chatbot_name;
          if (res.welcome_message) config.welcomeMessage = res.welcome_message;
          if (res.color) config.color = res.color;
          if (res.logo) config.logo = res.logo;
        }
      } catch (e) {}
      cb();
    };
    xhr.onerror = function () { cb(); };
    xhr.send();
  }

  loadConfig(render);
})();
