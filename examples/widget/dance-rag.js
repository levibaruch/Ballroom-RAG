/* Dance RAG demo widget: ask a question about ballroom/Latin technique, watch the steps, read the cited answer.
   Embed (one line each, in a project's Markdown):
     <link rel="stylesheet" href="/css/dance-rag.css">
     <div class="dance-rag" data-api="https://rag.example.com"></div>
     <script src="/js/dance-rag.js" defer></script>
   Talks to the API in dance-style-RAG (dance_rag_agentic/apps/api.py): POST /v1/ask streams server-sent events
   (steps, then the answer), GET /v1/examples, GET /v1/health. No dependencies. */
(function () {
  "use strict";

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // The answer text uses a small Markdown subset: **bold**, *italic*, "- " bullets, line breaks, footnotes [1, 2].
  function inline(text, prefix) {
    return escapeHtml(text)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
      .replace(/\[(\d+(?:,\s*\d+)*)\]/g, function (_, numbers) {
        return '<sup class="dr-fn">' + numbers.split(/,\s*/).map(function (n) {
          return '<a href="#' + prefix + "-src-" + n + '">' + n + "</a>";
        }).join(", ") + "</sup>";
      });
  }

  function renderAnswer(text, prefix) {
    var html = [], list = [];
    function flushList() {
      if (list.length) { html.push("<ul>" + list.join("") + "</ul>"); list = []; }
    }
    String(text).split("\n").forEach(function (raw) {
      var line = raw.trim();
      if (!line) { flushList(); return; }
      var bullet = line.match(/^[-*•]\s+(.*)$/);
      if (bullet) { list.push("<li>" + inline(bullet[1], prefix) + "</li>"); return; }
      flushList();
      var heading = line.match(/^#{1,4}\s+(.*)$/) || line.match(/^\*\*([^*]+)\*\*:?$/);
      if (heading) { html.push("<h4>" + inline(heading[1], prefix) + "</h4>"); return; }
      html.push("<p>" + inline(line, prefix) + "</p>");
    });
    flushList();
    return html.join("");
  }

  function setup(root, index) {
    var api = (root.getAttribute("data-api") || "").replace(/\/$/, "");
    var prefix = "dr" + index;
    root.innerHTML =
      '<form class="dr-form">' +
      '  <label class="dr-label" for="' + prefix + '-q">Ask a question about ballroom or Latin technique</label>' +
      '  <div class="dr-row">' +
      '    <input id="' + prefix + '-q" class="dr-input" type="text" maxlength="400" autocomplete="off"' +
      '           placeholder="e.g. How do I dance the Fan in Rumba?">' +
      '    <button class="dr-button" type="submit">Ask</button>' +
      "  </div>" +
      '  <div class="dr-examples" aria-label="Example questions"></div>' +
      "</form>" +
      '<ol class="dr-steps" aria-live="polite"></ol>' +
      '<div class="dr-answer" aria-live="polite"></div>' +
      '<section class="dr-trace" hidden><p class="dr-kicker">How the system answered <span class="dr-trace-sum"></span></p>' +
      '<p class="dr-legend"><span class="dr-cat-understand">understand</span><span class="dr-cat-search">search</span>' +
      '<span class="dr-cat-write">write</span><span class="dr-cat-check">check</span><span class="dr-cat-repair">repair</span>' +
      '<span class="dr-legend-hint">click a step to see what it decided</span></p>' +
      '<ol class="dr-nodes"></ol></section>' +
      '<p class="dr-note"></p>';

    var form = root.querySelector(".dr-form");
    var input = root.querySelector(".dr-input");
    var button = root.querySelector(".dr-button");
    var steps = root.querySelector(".dr-steps");
    var answer = root.querySelector(".dr-answer");
    var note = root.querySelector(".dr-note");
    var examples = root.querySelector(".dr-examples");
    var trace = root.querySelector(".dr-trace");
    var nodes = root.querySelector(".dr-nodes");
    var currentNode = null;

    // ---- the public trace: one collapsible block per graph step, filled with the whitelisted details the API sends.
    // The API sends text lines; here they are turned into facts, bars, tables and lists so they read easily.
    var VERDICT = { SUPPORTED: "supported", CONTRADICTED: "contradicted", NOT_IN_EVIDENCE: "not in evidence" };
    var LONG_TEXT = /^(answer \(raw|answer after the repairs|polished answer|repair decisions)/;
    var nodeCount = 0, totalSeconds = 0;
    var CATEGORY = {                     // colour per phase of the graph
      ExtractEntities: "understand", ClassifyKind: "understand", SplitQuestion: "understand",
      FigurePath: "search", ComparePath: "search", Background: "search", TermPath: "search", TechniqueAgent: "search",
      UnsureAgent: "search", FillGap: "search",
      Answer: "write", GeneralAdvice: "write", Polish: "write", AboutPath: "write",
      CheckCites: "check", JudgeClaims: "check",
      RepairClaims: "repair", Refuse: "repair"
    };

    function cleanValue(text) {          // "['Rumba', 'Jive']" -> "Rumba, Jive"; "None" / "[]" -> "-"
      var value = String(text).trim().replace(/^\[(.*)\]$/, "$1").replace(/'/g, "");
      return value === "" || value === "None" ? "-" : value;
    }
    function withoutIds(text) {          // the books' 6-digit unit ids mean nothing to a reader
      return String(text).replace(/\s*\[\d{6}(?:\.\d+)?(?:,\s*\d{6}(?:\.\d+)?)*\]/g, "").replace(/^\d{6}\s+/, "");
    }
    function bodyOf() { return currentNode && currentNode.querySelector(".dr-node-body"); }

    function startNode(event) {
      trace.hidden = false;
      nodeCount += 1;
      var item = document.createElement("li");
      if (currentNode) currentNode.classList.remove("dr-running");
      item.className = "dr-node dr-running dr-cat-" + (CATEGORY[event.node] || "finish");
      item.innerHTML = '<details><summary><span class="dr-node-num">' + nodeCount + "</span>" +
        '<span class="dr-node-title">' + escapeHtml(event.message) + "</span>" +
        '<span class="dr-node-time"></span><span class="dr-node-summary"></span></summary>' +
        '<div class="dr-node-body"><p class="dr-node-name">' + escapeHtml(event.node) + "</p></div></details>";
      nodes.appendChild(item);
      currentNode = item;
    }

    function finishNode(event) {
      if (!currentNode || event.node === "start") return;
      currentNode.classList.remove("dr-running");
      if (event.seconds != null) {
        totalSeconds += Number(event.seconds) || 0;
        currentNode.querySelector(".dr-node-time").textContent = event.seconds + " s";
      }
      currentNode.querySelector(".dr-node-summary").textContent = withoutIds(event.summary || "");
      trace.querySelector(".dr-trace-sum").textContent = "· " + nodeCount + " steps · " + Math.round(totalSeconds) + " s";
    }

    function addTool(event) {
      var body = bodyOf();
      if (!body) return;
      var list = body.querySelector(".dr-tools");
      if (!list) {
        body.insertAdjacentHTML("beforeend", '<p class="dr-detail-label">searches and reads</p><ul class="dr-tools"></ul>');
        list = body.querySelector(".dr-tools");
      }
      var call = String(event.call), text = call;
      var search = call.match(/^search\((.*)\)$/), read = call.match(/^read_section\(section_id=(\d+)\)$/);
      if (search) {
        var args = {};
        search[1].replace(/(\w+)=('([^']*)'|[^,]*)/g, function (_, key, raw, quoted) { args[key] = quoted != null ? quoted : raw; });
        text = "search “" + (args.query || "") + "”" + (args.dance ? " in " + args.dance : "") + (args.kind ? " (" + args.kind + ")" : "");
      } else if (read) {
        text = "read section " + read[1];
      }
      var item = document.createElement("li");
      item.textContent = text;
      list.appendChild(item);
    }

    function factsBlock(body) {          // consecutive short facts share one grid
      var last = body.lastElementChild;
      if (last && last.classList.contains("dr-facts")) return last;
      var grid = document.createElement("dl");
      grid.className = "dr-facts";
      body.appendChild(grid);
      return grid;
    }

    function asTable(lines) {            // column-aligned text lines -> a table
      var rows = lines.map(function (line) { return withoutIds(line).split(/\s{2,}/).filter(Boolean); });
      var width = Math.max.apply(null, rows.map(function (cells) { return cells.length; }));
      rows = rows.map(function (cells) {   // a short row (e.g. "total" with no model) lost empty cells after the first
        while (cells.length < width) cells.splice(1, 0, "");
        return cells;
      });
      var header = rows.length > 1 && width > 2 && rows[0].every(function (cell) { return !/\d/.test(cell); });
      return '<div class="dr-table-wrap"><table class="dr-table">' + rows.map(function (cells, r) {
        var tag = header && r === 0 ? "th" : "td";
        return "<tr>" + cells.map(function (cell) {
          var number = /^[\d.,$%]+$/.test(cell) ? ' class="dr-num"' : "";
          return "<" + tag + number + ">" + escapeHtml(cell) + "</" + tag + ">";
        }).join("") + "</tr>";
      }).join("") + "</table></div>";
    }

    function addDetail(event) {
      var body = bodyOf();
      if (!body) return;
      var label = String(event.label || ""), lines = event.lines || [], data = event.data || {};
      var html = "";
      var yesNo = label.match(/^(?:decision model:\s*)?(.*?)\s*P\(yes\)\s*([\d.]+).*?\((?:[a-z ]*?)([\d.]+)\)/i);

      if (data.kind === "probabilities") {
        html = '<p class="dr-detail-label">' + escapeHtml(label.replace(/^decision model:\s*/, "decision model · ")) + "</p>" +
          '<div class="dr-bars">' + data.items.map(function (item) {
            return '<div class="dr-bar' + (item[0] === data.marked ? " dr-marked" : "") + '"><span>' + escapeHtml(item[0]) +
                   '</span><i><b style="width:' + Math.round(item[1] * 100) + '%"></b></i><em>' + item[1].toFixed(2) + "</em></div>";
          }).join("") + "</div>";
      } else if (data.kind === "claims") {
        var good = data.items.filter(function (claim) { return claim.verdict ? claim.verdict === "SUPPORTED" && claim.p >= data.threshold : claim.p >= data.threshold; }).length;
        html = '<p class="dr-detail-label">' + escapeHtml(label) + " · " + good + " of " + data.items.length + " supported</p>" +
          '<ul class="dr-claims">' + data.items.map(function (claim) {
            var ok = claim.verdict ? claim.verdict === "SUPPORTED" && claim.p >= data.threshold : claim.p >= data.threshold;
            return '<li class="' + (ok ? "dr-ok" : "dr-no") + '"><span class="dr-verdict">' +
                   escapeHtml(VERDICT[claim.verdict] || (ok ? "supported" : "not supported")) +
                   (claim.p != null ? " " + Number(claim.p).toFixed(2) : "") + "</span><span>" + escapeHtml(claim.claim) + "</span></li>";
          }).join("") + "</ul>";
      } else if (yesNo) {
        var p = Number(yesNo[2]), cut = Number(yesNo[3]);
        html = '<p class="dr-detail-label">decision model · ' + escapeHtml(yesNo[1]) + "</p>" +
          '<div class="dr-bars"><div class="dr-bar' + (p >= cut ? " dr-marked" : "") + '"><span>P(yes)</span>' +
          '<i><b style="width:' + Math.round(p * 100) + '%"></b><u style="left:' + Math.round(cut * 100) + '%" title="threshold"></u></i>' +
          "<em>" + p.toFixed(2) + "</em></div></div>";
      } else if (!lines.length) {
        // "dances   ['Rumba']" (aligned) / "aspect used: lead" (colon) / "route FIGURE" (first word)
        var fact = label.match(/^(\S.*?)\s{2,}(.*)$/) || label.match(/^([^:]+?):\s+(.*)$/) ||
                   label.match(/^(\S+)\s+(.*)$/) || [null, label, ""];
        factsBlock(body).insertAdjacentHTML("beforeend",
          "<dt>" + escapeHtml(fact[1]) + "</dt><dd>" + escapeHtml(cleanValue(withoutIds(fact[2]))) + "</dd>");
        return;
      } else if (LONG_TEXT.test(label)) {
        html = '<details class="dr-text"><summary>' + escapeHtml(label) + "</summary><p>" +
               escapeHtml(lines.map(withoutIds).join("\n")) + "</p></details>";
      } else if (lines.filter(function (line) { return /\S\s{2,}\S/.test(line); }).length >= Math.max(1, lines.length / 2)) {
        html = '<p class="dr-detail-label">' + escapeHtml(label) + "</p>" + asTable(lines);
      } else if (lines.length > 6 && lines.every(function (line) { return line.length < 40; })) {
        html = '<p class="dr-detail-label">' + escapeHtml(label) + '</p><ul class="dr-chips">' + lines.map(function (line) {
          return "<li>" + escapeHtml(line) + "</li>";
        }).join("") + "</ul>";
      } else {
        html = '<p class="dr-detail-label">' + escapeHtml(label) + '</p><ul class="dr-lines">' + lines.map(function (line) {
          return "<li>" + escapeHtml(withoutIds(line).replace(/^-\s+/, "")) + "</li>";
        }).join("") + "</ul>";
      }
      body.insertAdjacentHTML("beforeend", '<div class="dr-detail">' + html + "</div>");
    }

    fetch(api + "/v1/examples").then(function (r) { return r.json(); }).then(function (data) {
      (data.examples || []).forEach(function (question) {
        var chip = document.createElement("button");
        chip.type = "button";
        chip.className = "dr-chip";
        chip.textContent = question;
        chip.addEventListener("click", function () { input.value = question; ask(question); });
        examples.appendChild(chip);
      });
    }).catch(function () {
      note.textContent = "The demo server is not reachable right now.";
    });

    function addStep(text, state) {
      var last = steps.lastElementChild;
      if (last && last.textContent === text) return;
      if (last) last.classList.remove("dr-active");
      var item = document.createElement("li");
      item.textContent = text;
      if (state) item.classList.add(state);
      steps.appendChild(item);
    }

    function showAnswer(event) {
      var parts = ['<div class="dr-body">' + renderAnswer(event.body || "", prefix) + "</div>"];
      if (event.tips) {
        parts.push('<aside class="dr-tips"><p class="dr-kicker">' + escapeHtml(event.tips_heading || "General advice, not from the books") + "</p>" +
                   renderAnswer(event.tips, prefix) + "</aside>");
      }
      if (event.sources && event.sources.length) {
        parts.push('<p class="dr-kicker">Sources</p><ol class="dr-sources">' + event.sources.map(function (s) {
          return '<li id="' + prefix + "-src-" + s.number + '" value="' + s.number + '">' +
                 '<span class="dr-where">' + escapeHtml(s.book) + ", page " + escapeHtml(s.page) +
                 (s.section ? " · " + escapeHtml(s.section) : "") + "</span>" +
                 (s.quote ? '<q>' + escapeHtml(s.quote) + "</q>" : "") + "</li>";
        }).join("") + "</ol>");
      }
      answer.innerHTML = parts.join("");
    }

    function ask(question) {
      question = (question || "").trim();
      if (!question || button.disabled) return;
      button.disabled = true;
      steps.innerHTML = "";
      answer.innerHTML = "";
      nodes.innerHTML = "";
      trace.hidden = true;
      trace.querySelector(".dr-trace-sum").textContent = "";
      steps.hidden = false;
      currentNode = null;
      nodeCount = 0;
      totalSeconds = 0;
      note.textContent = "Answers take 30 to 60 seconds: every claim is checked against the books.";
      addStep("Sending the question", "dr-active");

      fetch(api + "/v1/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question })
      }).then(function (response) {
        if (!response.ok) {
          return response.json().catch(function () { return {}; }).then(function (data) {
            throw new Error(data.error || "The demo server answered with an error (" + response.status + ").");
          });
        }
        var reader = response.body.getReader();
        var decoder = new TextDecoder();
        var buffer = "";
        function read() {
          return reader.read().then(function (chunk) {
            if (chunk.done) return;
            buffer += decoder.decode(chunk.value, { stream: true });
            var events = buffer.split("\n\n");
            buffer = events.pop();
            events.forEach(function (block) {
              var line = block.split("\n").filter(function (l) { return l.indexOf("data: ") === 0; })[0];
              if (!line) return;
              var event = JSON.parse(line.slice(6));
              if (event.type === "node") { steps.hidden = true; startNode(event); }
              else if (event.type === "step") finishNode(event);
              else if (event.type === "detail") addDetail(event);
              else if (event.type === "tool") addTool(event);
              else if (event.type === "answer") { showAnswer(event); steps.hidden = true; note.textContent = ""; }
              else if (event.type === "error") throw new Error(event.message);
            });
            return read();
          });
        }
        return read();
      }).catch(function (error) {
        steps.hidden = false;
        if (currentNode) currentNode.classList.remove("dr-running");
        addStep(error.message || "Something went wrong.", "dr-error");
        note.textContent = "";
      }).then(function () {
        button.disabled = false;
        var last = steps.lastElementChild;
        if (last) last.classList.remove("dr-active");
      });
    }

    form.addEventListener("submit", function (event) { event.preventDefault(); ask(input.value); });
  }

  function start() {
    Array.prototype.forEach.call(document.querySelectorAll(".dance-rag[data-api]"), setup);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
