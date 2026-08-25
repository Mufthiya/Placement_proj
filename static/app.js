document.addEventListener("DOMContentLoaded", () => {
  const feelBtn = document.getElementById("shareBtn");
  if (feelBtn) {
    feelBtn.addEventListener("click", async () => {
      const box = document.getElementById("feelResult");
      const text = document.getElementById("feelText").value.trim();
      if (!text) { alert("Type a sentence about how you feel first."); return; }
      feelBtn.disabled = true;
      try {
        const res = await fetch("/api/assess", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode: "text", text }),
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        let html = "<b>" + data.top_condition + " · " + data.confidence + "%</b>"
                 + "<span>" + data.message + "</span>";
        if (data.go === "mental") {
          html += '<br><a class="row-link" href="/checkin/mental">Run the mental check-in &rarr;</a>';
        }
        box.innerHTML = html;
        box.classList.remove("hidden");
      } catch (err) {
        alert("Could not analyze: " + err.message);
      } finally {
        feelBtn.disabled = false;
      }
    });
  }

  const submitBtn = document.getElementById("submitBtn");
  if (!submitBtn) return;

  submitBtn.addEventListener("click", async () => {
    if (!window.MODELS_READY) {
      alert("Models are not loaded. Run: python main.py");
      return;
    }
    const form = document.getElementById("checkinForm");
    const section = form.dataset.section;
    const answers = {};
    let missing = [];

    document.querySelectorAll(".choice-row").forEach((group) => {
      const q = group.dataset.group;
      const multi = group.dataset.multi === "true";
      if (multi) {
        const vals = [...group.querySelectorAll("input:checked")].map((i) => i.value);
        if (!vals.length) missing.push(q);
        answers[q] = vals;
      } else {
        const sel = group.querySelector("input:checked");
        if (!sel) { missing.push(q); return; }
        answers[q] = sel.value;
      }
    });

    ["name", "disease", "age"].forEach((id) => {
      const el = document.getElementById("f_" + id);
      if (!el) return;
      if (!el.value.trim()) {
        if (id !== "name") missing.push(id);
        else answers[id] = "Guest";
      } else {
        answers[id] = el.value.trim();
      }
    });

    if (missing.length) {
      alert("Please answer: " + missing.join(", "));
      return;
    }

    document.getElementById("overlay").classList.remove("hidden");

    try {
      const res = await fetch("/api/assess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [section]: answers }),
      });
      const data = await res.json();
      if (data.redirect) window.location.href = data.redirect;
      else throw new Error(data.error || "Unknown error");
    } catch (err) {
      alert("Assessment failed: " + err.message);
      document.getElementById("overlay").classList.add("hidden");
    }
  });

  const bookBtn = document.getElementById("bookBtn");
  if (bookBtn) {
    bookBtn.addEventListener("click", async () => {
      const modeSel = document.querySelector('input[name="q_mode"]:checked');
      const payload = {
        name: document.getElementById("b_name").value.trim() || "Guest",
        doctor: document.getElementById("b_doctor").value,
        date: document.getElementById("b_date").value,
        time: document.getElementById("b_time").value,
        mode: modeSel ? modeSel.value : "",
        contact: document.getElementById("b_contact").value.trim(),
        criticality: window.PATIENT_ALERT || "",
        reason: window.PATIENT_ISSUE || "",
      };
      if (!payload.date || !payload.time || !payload.mode) {
        alert("Pick a date, time and consultation mode.");
        return;
      }
      bookBtn.disabled = true;
      try {
        const res = await fetch("/api/book", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || "Booking failed");
        const b = data.booking;
        const box = document.getElementById("bookResult");
        box.innerHTML = "<b>Appointment confirmed.</b> " + b.doctor + " will see "
          + b.name + " on " + b.date + " at " + b.time + " (" + b.mode
          + "). Booking ID: " + b.id;
        box.classList.remove("hidden");
      } catch (err) {
        alert("Booking failed: " + err.message);
      } finally {
        bookBtn.disabled = false;
      }
    });
  }
});
