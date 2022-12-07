function message(status, shake=false, id="") {
    if (shake) {
      $("#"+id).effect("shake", {direction: "right", times: 2, distance: 8}, 250);
    } 
    document.getElementById("feedback").innerHTML = status;
    $("#feedback").show().delay(2000).fadeOut();
  }
  
  function error(type) {
    $("."+type).css("border-color", "#E14448");
  }
  
  var login = function() {
    $.post({
      type: "POST",
      url: "/",
      data: {"username": $("#login-user").val(), 
             "password": $("#login-pass").val()},
      success(response){
        var status = JSON.parse(response)["status"];
        if (status === "Login successful") { location.reload(); }
        else { error("login-input"); }
      }
    });
  };
  
  $(document).ready(function() {
    
    $(document).on("click", "#login-button", login);
    $(document).keypress(function(e) {if(e.which === 13) {login();}});
    
    $(document).on("click", "#signup-button", function() {
      $.post({
        type: "POST",
        url: "/signup",
        data: {"username": $("#signup-user").val(), 
               "password": $("#signup-pass").val(), 
               "email": $("#signup-mail").val()},
        success(response) {
          var status = JSON.parse(response)["status"];
          if (status === "Signup successful") { location.reload(); }
          else { message(status, true, "signup-box"); }
        }
      });
    });
  
    $(document).on("click", "#save", function() {
      $.post({
        type: "POST",
        url: "/settings",
        data: {"username": $("#settings-user").val(), 
               "password": $("#settings-pass").val(), 
               "email": $("#settings-mail").val()},
        success(response){
          message(JSON.parse(response)["status"]);
        }
      });
    });
  });
  
  // Open or Close mobile & tablet menu
  $("#navbar-burger-id").click(function () {
    if($("#navbar-burger-id").hasClass("is-active")){
      $("#navbar-burger-id").removeClass("is-active");
      $("#navbar-menu-id").removeClass("is-active");
    }else {
      $("#navbar-burger-id").addClass("is-active");
      $("#navbar-menu-id").addClass("is-active");
    }
  });

$(document).ready(function(){
    $(".push_menu").click(function(){
         $(".wrapper").toggleClass("active");
    });
});

// Pagination Demonstration


$(document).ready(function(){
  $('#table_pagination').dataTable({
    ordering: false,
  });
});

$(document).ready(function(){
  $('#table_pagination2').dataTable({
    ordering: false,
  });
});

$(document).ready(function(){
  $('#table_pagination3').dataTable({
    ordering: false,
  });
});

// Modal Detail IG
var btn = document.querySelector('#showModal');
var modalDlg = document.querySelector('#image-modal');
var imageModalCloseBtn = document.querySelector('#image-modal-close');
btn.addEventListener('click', function(){
  modalDlg.classList.add('is-active');
});

imageModalCloseBtn.addEventListener('click', function(){
  modalDlg.classList.remove('is-active');
});

// Mini tabs in Demonstration
function switchDataAwal() {
  removeActive();
  hideAll();
  $("#data-awal").addClass("is-active");
  $("#data-awal-content").removeClass("is-hidden");
}

function switchDetailIG() {
  removeActive();
  hideAll();
  $("#detail-ig").addClass("is-active");
  $("#detail-ig-content").removeClass("is-hidden");
}

function switchHasilIG() {
  removeActive();
  hideAll();
  $("#hasil-ig").addClass("is-active");
  $("#hasil-ig-content").removeClass("is-hidden");
}

function switchHasilLDA() {
  removeActive();
  hideAll();
  $("#hasil-lda-tab").addClass("is-active");
  $("#hasil-lda-content").removeClass("is-hidden");
}

function switchHasilSVM() {
  removeActive();
  hideAll();
  $("#hasil-svm").addClass("is-active");
  $("#hasil-svm-content").removeClass("is-hidden");
}

function removeActive() {
  $("li").each(function() {
    $(this).removeClass("is-active");
  });
}

function hideAll(){
  $("#data-awal-content").addClass("is-hidden");
  $("#hasil-lda-content").addClass("is-hidden");
  $("#detail-ig-content").addClass("is-hidden");
  $("#hasil-ig-content").addClass("is-hidden");
  $("#hasil-svm-content").addClass("is-hidden");
}

// Download Crawling data
function downloadCSV(csv, filename) {
  var csvFile;
  var downloadLink;

  // CSV file
  csvFile = new Blob([csv], {type: "text/csv"});

  // Download link
  downloadLink = document.createElement("a");

  // File name
  downloadLink.download = filename;

  // Create a link to the file
  downloadLink.href = window.URL.createObjectURL(csvFile);

  // Hide download link
  downloadLink.style.display = "none";

  // Add the link to DOM
  document.body.appendChild(downloadLink);

  // Click download link
  downloadLink.click();
}

function exportTableToCSV(filename) {
  var csv = [];
  var rows = document.querySelectorAll("table tr");
  
  for (var i = 0; i < rows.length; i++) {
      var row = [], cols = rows[i].querySelectorAll("td, th");
      
      for (var j = 0; j < cols.length; j++) {
          // Clean innertext to remove multiple spaces and jumpline (break csv)
          var data = cols[j].innerText.replace(/(\r\n|\n|\r)/gm, '').replace(/(\s\s)/gm, ' ')
          // Escape double-quote with double-double-quote (see https://stackoverflow.com/questions/17808511/properly-escape-a-double-quote-in-csv)
          data = data.replace(/"/g, '""');
          // Push escaped string
          row.push('"' + data + '"');
      }
      csv.push(row.join(","));        
  }

  // Download CSV file
  downloadCSV(csv.join("\n"), filename);
}