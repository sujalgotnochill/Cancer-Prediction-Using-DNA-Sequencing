/**
 * DNA Sequence Classifier — Interactive Client-side Script
 * Vanilla JavaScript (No external frameworks)
 */

document.addEventListener("DOMContentLoaded", function () {
  // ------------------------------------------------------------------------
  // Elements
  // ------------------------------------------------------------------------
  const sequenceTextarea = document.getElementById("sequence_text");
  const charCounter = document.getElementById("char-counter");
  const liveValidation = document.getElementById("live-validation");
  const fileInput = document.getElementById("sequence_file");
  const uploadZone = document.getElementById("file-upload-zone");
  const selectedFileBadge = document.getElementById("selected-file-badge");
  const selectedFileName = document.getElementById("selected-file-name");
  const clearFileBtn = document.getElementById("clear-file-btn");
  const sampleBtn = document.getElementById("btn-sample-sequence");
  const clearBtn = document.getElementById("btn-clear-input");
  const predictForm = document.getElementById("predict-form");
  const submitBtn = document.getElementById("btn-submit-predict");

  const MAX_LENGTH = 50000;
  const ALLOWED_BASES = new Set(["A", "C", "G", "T", "N"]);

  // ------------------------------------------------------------------------
  // Character Count & Real-time Validation
  // ------------------------------------------------------------------------
  function updateSequenceMetrics() {
    if (!sequenceTextarea || !charCounter) return;

    const raw = sequenceTextarea.value;
    if (!raw.trim()) {
      charCounter.textContent = `0 / ${MAX_LENGTH.toLocaleString()} bp`;
      charCounter.className = "char-counter";
      if (liveValidation) {
        liveValidation.className = "live-validation-box";
        liveValidation.textContent = "";
      }
      return;
    }

    // Strip FASTA header line if present and ignore whitespace
    const lines = raw.split("\n");
    let seqOnly = "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith(">")) {
        seqOnly += trimmed.replace(/\s+/g, "");
      }
    }

    const len = seqOnly.length;
    charCounter.textContent = `${len.toLocaleString()} / ${MAX_LENGTH.toLocaleString()} bp`;

    // Check invalid characters
    const upper = seqOnly.toUpperCase();
    const invalidSet = new Set();
    for (let i = 0; i < upper.length; i++) {
      const char = upper[i];
      if (!ALLOWED_BASES.has(char)) {
        invalidSet.add(char);
      }
    }

    if (liveValidation) {
      if (invalidSet.size > 0) {
        const invalidList = Array.from(invalidSet).slice(0, 6).join(", ");
        liveValidation.textContent = `Warning: Unsupported nucleotide character(s) detected: [${invalidList}]. Only A, C, G, T, and N are accepted.`;
        liveValidation.className = "live-validation-box error";
      } else if (len > MAX_LENGTH) {
        liveValidation.textContent = `Warning: Sequence length (${len.toLocaleString()} bp) exceeds maximum permitted length of ${MAX_LENGTH.toLocaleString()} bp.`;
        liveValidation.className = "live-validation-box error";
        charCounter.className = "char-counter limit-exceeded";
      } else if (len > 0 && len < 4) {
        liveValidation.textContent = `Note: Sequence is shorter than 4 bp (minimum required for 4-mer feature extraction).`;
        liveValidation.className = "live-validation-box error";
      } else {
        liveValidation.textContent = "";
        liveValidation.className = "live-validation-box";
        charCounter.className = len > MAX_LENGTH * 0.9 ? "char-counter limit-warning" : "char-counter";
      }
    }
  }

  if (sequenceTextarea) {
    sequenceTextarea.addEventListener("input", updateSequenceMetrics);
    // Initial evaluation
    updateSequenceMetrics();
  }

  // ------------------------------------------------------------------------
  // Load Demonstration Sequence
  // ------------------------------------------------------------------------
  if (sampleBtn && sequenceTextarea) {
    sampleBtn.addEventListener("click", function () {
      fetch("/api/sample")
        .then((res) => res.json())
        .then((data) => {
          if (data && data.fasta) {
            sequenceTextarea.value = data.fasta;
            clearSelectedFile();
            updateSequenceMetrics();
            sequenceTextarea.focus();
          }
        })
        .catch((err) => {
          console.warn("Could not load sample sequence via API:", err);
          // Fallback static demonstration string
          sequenceTextarea.value =
            ">synthetic_demo_sequence_educational_use_only | Synthetic benchmark test construct\n" +
            "ATGGCGTACGTTAGCTAGCTAGCTGATCGATCGATCGTAGCTAGCTAGCTAGCTAGCTA\n" +
            "GCTAGCGATCGAATCGATCGATCGATCGATCGATCGATCGATCGATCGTAGCTAGCTAG\n" +
            "CTAAGCTAGCTAGCTAGCTAGCGATCGAATCGATCGATCGATCGATCGATCGATCGATC\n" +
            "GATCGATCGTAGCTAGCTAGCTAAGCTAGCTAGCTAGCTAGCGATCGAATCGATCGATC\n" +
            "GATCGATCGATCGATCGATCGATCGATCGTAGCTAGCTAGCTAAGCTAGCTAGCTAGCT";
          clearSelectedFile();
          updateSequenceMetrics();
        });
    });
  }

  // ------------------------------------------------------------------------
  // Clear Input Button
  // ------------------------------------------------------------------------
  if (clearBtn && sequenceTextarea) {
    clearBtn.addEventListener("click", function () {
      sequenceTextarea.value = "";
      clearSelectedFile();
      updateSequenceMetrics();
      sequenceTextarea.focus();
    });
  }

  // ------------------------------------------------------------------------
  // Drag & Drop / File Upload
  // ------------------------------------------------------------------------
  function clearSelectedFile() {
    if (fileInput) fileInput.value = "";
    if (selectedFileBadge) selectedFileBadge.style.display = "none";
    if (selectedFileName) selectedFileName.textContent = "";
  }

  if (clearFileBtn) {
    clearFileBtn.addEventListener("click", function (e) {
      e.stopPropagation();
      clearSelectedFile();
    });
  }

  if (uploadZone && fileInput) {
    uploadZone.addEventListener("click", function () {
      fileInput.click();
    });

    fileInput.addEventListener("change", function () {
      if (fileInput.files && fileInput.files.length > 0) {
        handleFileSelect(fileInput.files[0]);
      }
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      uploadZone.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        uploadZone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      uploadZone.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        uploadZone.classList.remove("dragover");
      });
    });

    uploadZone.addEventListener("drop", function (e) {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        fileInput.files = dt.files;
        handleFileSelect(dt.files[0]);
      }
    });
  }

  function handleFileSelect(file) {
    if (!file) return;

    const allowed = [".txt", ".fasta", ".fa", ".fna"];
    const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
    if (!allowed.includes(ext)) {
      alert(`Unsupported file format '${file.name}'. Please choose a .fasta, .fa, .fna, or .txt file.`);
      clearSelectedFile();
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      alert(`File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum file size is 5 MB.`);
      clearSelectedFile();
      return;
    }

    if (selectedFileBadge && selectedFileName) {
      selectedFileName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      selectedFileBadge.style.display = "inline-flex";
    }

    // Read file text to preview in textarea
    const reader = new FileReader();
    reader.onload = function (e) {
      if (sequenceTextarea && e.target && typeof e.target.result === "string") {
        sequenceTextarea.value = e.target.result;
        updateSequenceMetrics();
      }
    };
    reader.readAsText(file);
  }

  // ------------------------------------------------------------------------
  // Form Submit Loading State
  // ------------------------------------------------------------------------
  if (predictForm && submitBtn) {
    predictForm.addEventListener("submit", function (e) {
      const hasText = sequenceTextarea && sequenceTextarea.value.trim().length > 0;
      const hasFile = fileInput && fileInput.files && fileInput.files.length > 0;

      if (!hasText && !hasFile) {
        e.preventDefault();
        alert("Please enter a DNA sequence or select a FASTA file before submitting.");
        return;
      }

      submitBtn.disabled = true;
      submitBtn.innerHTML = `
        <svg class="spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation: spin 1s linear infinite;">
          <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
          <path d="M12 2a10 10 0 0 1 10 10"></path>
        </svg>
        <span>Running Inference...</span>
      `;
    });
  }

  // ------------------------------------------------------------------------
  // Modal Handling (Confirmation Dialog for Clear History)
  // ------------------------------------------------------------------------
  const clearHistoryTrigger = document.getElementById("btn-open-clear-modal");
  const clearModal = document.getElementById("clear-history-modal");
  const cancelModalBtn = document.getElementById("btn-cancel-modal");

  if (clearHistoryTrigger && clearModal) {
    clearHistoryTrigger.addEventListener("click", function () {
      clearModal.classList.add("open");
    });
  }

  if (cancelModalBtn && clearModal) {
    cancelModalBtn.addEventListener("click", function () {
      clearModal.classList.remove("open");
    });
  }

  if (clearModal) {
    clearModal.addEventListener("click", function (e) {
      if (e.target === clearModal) {
        clearModal.classList.remove("open");
      }
    });
  }
});

// CSS keyframe for spinner
const styleSheet = document.createElement("style");
styleSheet.textContent = `@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`;
document.head.appendChild(styleSheet);

