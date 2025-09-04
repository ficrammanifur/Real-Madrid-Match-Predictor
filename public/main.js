// main.js
class RealMadridPredictor {
  constructor() {
    this.form = document.getElementById("predictionForm");
    this.resultsContainer = document.getElementById("results");
    this.predictBtn = this.form.querySelector(".predict-btn");
    this.btnText = this.predictBtn.querySelector(".btn-text");
    this.loadingSpinner = this.predictBtn.querySelector(".loading-spinner");
    this.recentMatchesContainer = document.getElementById("recentMatches");
    this.upcomingMatchesContainer = document.getElementById("upcomingMatches");
    this.recentDaysSelect = document.getElementById("recentDays");
    this.upcomingDaysSelect = document.getElementById("upcomingDays");
    this.refreshRecentBtn = document.getElementById("refreshRecent");
    this.refreshUpcomingBtn = document.getElementById("refreshUpcoming");
    this.chart = null; // Store Chart.js instance

    this.init();
  }

  init() {
    this.form.addEventListener("submit", (e) => this.handleSubmit(e));
    this.refreshRecentBtn.addEventListener("click", () => this.fetchRecentMatches());
    this.refreshUpcomingBtn.addEventListener("click", () => this.fetchUpcomingMatches());
    this.recentDaysSelect.addEventListener("change", () => this.fetchRecentMatches());
    this.upcomingDaysSelect.addEventListener("change", () => this.fetchUpcomingMatches());
    this.addInputValidation();
    this.setupTabs();
    this.fetchRecentMatches();
    this.fetchUpcomingMatches();
    this.fetchTeamSuggestions();
  }

  setupTabs() {
    const tabs = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        tabs.forEach((t) => t.classList.remove("active"));
        tabContents.forEach((c) => c.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(`${tab.dataset.tab}-tab`).classList.add("active");
      });
    });
  }

  addInputValidation() {
    const inputs = this.form.querySelectorAll("input, select");
    inputs.forEach((input) => {
      input.addEventListener("input", () => this.validateForm());
    });
  }

  validateForm() {
    const opponent = document.getElementById("opponent").value.trim();
    const competition = document.getElementById("competition").value;
    const venue = document.getElementById("venue").value;
    const isValid = opponent && competition && venue;
    this.predictBtn.disabled = !isValid;
    return isValid;
  }

  async handleSubmit(e) {
    e.preventDefault();
    if (!this.validateForm()) {
      this.showError("Mohon lengkapi semua field yang diperlukan.");
      return;
    }

    const formData = new FormData(this.form);
    const data = {
      opponent: formData.get("opponent").trim(),
      competition: formData.get("competition"),
      venue: formData.get("venue"),
      madridForm: parseFloat(formData.get("madridForm")),
      madridXg: parseFloat(formData.get("madridXg")),
      madridConcede: parseFloat(formData.get("madridConcede")),
      opponentForm: parseFloat(formData.get("opponentForm")),
      restDays: parseInt(formData.get("restDays")),
      keyPlayersOut: parseInt(formData.get("keyPlayersOut"))
    };

    this.setLoading(true);
    try {
      const prediction = await this.getPrediction(data);
      this.displayResults(prediction);
    } catch (error) {
      console.error("Prediction error:", error);
      this.showError("Terjadi kesalahan saat melakukan prediksi. Silakan coba lagi.");
    } finally {
      this.setLoading(false);
    }
  }

  async getPrediction(data) {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  }

  displayResults(prediction) {
    document.getElementById("winPercentage").textContent = `${prediction.win}%`;
    document.getElementById("drawPercentage").textContent = `${prediction.draw}%`;
    document.getElementById("lossPercentage").textContent = `${prediction.loss}%`;
    document.getElementById("confidence").textContent = `${prediction.confidence}%`;
    document.getElementById("modelUsed").textContent = prediction.model_used;
    this.resultsContainer.style.display = "block";
    this.resultsContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
    this.animateCounters(prediction);
    this.displayPredictionChart(prediction);
  }

  animateCounters(prediction) {
    const counters = [
      { element: document.getElementById("winPercentage"), target: prediction.win },
      { element: document.getElementById("drawPercentage"), target: prediction.draw },
      { element: document.getElementById("lossPercentage"), target: prediction.loss },
      { element: document.getElementById("confidence"), target: prediction.confidence }
    ];
    counters.forEach(({ element, target }) => this.animateCounter(element, target));
  }

  animateCounter(element, target) {
    let current = 0;
    const increment = target / 30;
    const timer = setInterval(() => {
      current += increment;
      if (current >= target) {
        current = target;
        clearInterval(timer);
      }
      element.textContent = `${Math.round(current)}%`;
    }, 50);
  }

  displayPredictionChart(prediction) {
    const ctx = document.getElementById("predictionChart").getContext("2d");
    if (this.chart) {
      this.chart.destroy(); // Destroy previous chart to prevent overlap
    }
    this.chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Menang', 'Seri', 'Kalah'],
        datasets: [{
          label: 'Probabilitas Prediksi',
          data: [prediction.win, prediction.draw, prediction.loss],
          backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
          borderColor: ['#065f46', '#b45309', '#b91c1c'],
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            title: {
              display: true,
              text: 'Probabilitas (%)'
            }
          },
          x: {
            title: {
              display: true,
              text: 'Hasil'
            }
          }
        },
        plugins: {
          legend: {
            display: false
          },
          title: {
            display: true,
            text: 'Prediksi Hasil Pertandingan'
          }
        }
      }
    });
  }

  setLoading(isLoading) {
    this.predictBtn.disabled = isLoading;
    this.btnText.textContent = isLoading ? "Memproses..." : "Prediksi Hasil";
    this.loadingSpinner.style.display = isLoading ? "block" : "none";
  }

  showError(message) {
    let errorDiv = document.querySelector(".error-message");
    if (!errorDiv) {
      errorDiv = document.createElement("div");
      errorDiv.className = "error-message";
      errorDiv.style.cssText = `
        background: #fee2e2;
        color: #dc2626;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
        border: 1px solid #fecaca;
        text-align: center;
        font-weight: 500;
      `;
      this.form.appendChild(errorDiv);
    }
    errorDiv.textContent = message;
    setTimeout(() => errorDiv.remove(), 5000);
  }

  async fetchTeamSuggestions() {
    try {
      const response = await fetch("/api/team-list");
      const data = await response.json();
      const datalist = document.getElementById("teamSuggestions");
      datalist.innerHTML = "";
      data.teams.forEach((team) => {
        const option = document.createElement("option");
        option.value = team;
        datalist.appendChild(option);
      });
    } catch (error) {
      console.error("Error fetching team suggestions:", error);
    }
  }

  async fetchRecentMatches() {
    this.recentMatchesContainer.innerHTML = "<div class='loading-message'>Memuat data pertandingan...</div>";
    try {
      const days = this.recentDaysSelect.value;
      const response = await fetch(`/api/recent-matches?days=${days}`);
      const data = await response.json();
      if (data.error) {
        this.recentMatchesContainer.innerHTML = "<div class='error-message'>Gagal memuat data pertandingan.</div>";
        return;
      }
      this.displayMatches(this.recentMatchesContainer, data.matches, true);
    } catch (error) {
      console.error("Error fetching recent matches:", error);
      this.recentMatchesContainer.innerHTML = "<div class='error-message'>Gagal memuat data pertandingan.</div>";
    }
  }

  async fetchUpcomingMatches() {
    this.upcomingMatchesContainer.innerHTML = "<div class='loading-message'>Memuat jadwal pertandingan...</div>";
    try {
      const days = this.upcomingDaysSelect.value;
      const response = await fetch(`/api/upcoming-matches?days=${days}`);
      const data = await response.json();
      if (data.error) {
        this.upcomingMatchesContainer.innerHTML = "<div class='error-message'>Gagal memuat jadwal pertandingan.</div>";
        return;
      }
      this.displayMatches(this.upcomingMatchesContainer, data.matches, false);
    } catch (error) {
      console.error("Error fetching upcoming matches:", error);
      this.upcomingMatchesContainer.innerHTML = "<div class='error-message'>Gagal memuat jadwal pertandingan.</div>";
    }
  }

  displayMatches(container, matches, isRecent) {
    container.innerHTML = "";
    if (matches.length === 0) {
      container.innerHTML = "<div class='no-data'>Tidak ada pertandingan ditemukan.</div>";
      return;
    }
    matches.forEach((match) => {
      const matchDiv = document.createElement("div");
      matchDiv.className = "match-item";
      if (isRecent) {
        matchDiv.innerHTML = `
          <div class="match-date">${match.date}</div>
          <div class="match-details">
            <span class="match-opponent">${match.opponent}</span>
            <span class="match-competition">(${match.competition}, ${match.venue})</span>
            <span class="match-score">${match.real_madrid_score} - ${match.opponent_score} (${match.result})</span>
            ${match.goal_scorers.length > 0 ? `
              <div class="goal-scorers">
                <strong>Pencetak Gol:</strong>
                ${match.goal_scorers.map(g => `${g.player} (${g.minute}'${g.assist ? `, Assist: ${g.assist}` : ''})`).join(", ")}
              </div>
            ` : ""}
          </div>
        `;
      } else {
        matchDiv.innerHTML = `
          <div class="match-date">${match.date} ${match.time}</div>
          <div class="match-details">
            <span class="match-opponent">${match.opponent}</span>
            <span class="match-competition">(${match.competition}, ${match.venue})</span>
          </div>
        `;
      }
      container.appendChild(matchDiv);
    });
  }
}

class UIEnhancements {
  static init() {
    this.addFormEnhancements();
    this.addKeyboardShortcuts();
    this.addTooltips();
  }

  static addFormEnhancements() {
    const opponentInput = document.getElementById("opponent");
    opponentInput.addEventListener("input", (e) => {
      const words = e.target.value.split(" ");
      const capitalizedWords = words.map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase());
      e.target.value = capitalizedWords.join(" ");
    });
  }

  static addKeyboardShortcuts() {
    document.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && e.target.tagName !== "BUTTON") {
        const form = document.getElementById("predictionForm");
        const submitBtn = form.querySelector(".predict-btn");
        if (!submitBtn.disabled) {
          submitBtn.click();
        }
      }
    });
  }

  static addTooltips() {
    const tooltips = {
      opponent: "Masukkan nama tim lawan yang akan dihadapi Real Madrid",
      competition: "Pilih kompetisi dimana pertandingan akan berlangsung",
      venue: "Pilih apakah Real Madrid bermain di kandang atau tandang",
      madridForm: "Rata-rata poin per pertandingan Real Madrid (0-3)",
      madridXg: "Rata-rata expected goals (xG) Real Madrid",
      madridConcede: "Rata-rata gol kebobolan per pertandingan",
      opponentForm: "Rata-rata poin per pertandingan lawan (0-3)",
      restDays: "Jumlah hari istirahat sebelum pertandingan",
      keyPlayersOut: "Jumlah pemain kunci Real Madrid yang absen"
    };
    Object.entries(tooltips).forEach(([id, text]) => {
      const element = document.getElementById(id);
      if (element) element.title = text;
    });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new RealMadridPredictor();
  UIEnhancements.init();
  console.log("Real Madrid Match Predictor initialized successfully!");
});