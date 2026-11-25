const form = document.getElementById("signupForm");
const message = document.getElementById("message");
function openDashboard(name){localStorage.setItem("agriguard_user_name",name||"User");location.href="dashboard.html";}
form?.addEventListener("submit",e=>{e.preventDefault();const name=document.getElementById("name").value.trim();const p=document.getElementById("password").value;const c=document.getElementById("confirmPassword").value;if(p!==c){message.textContent="Passwords do not match.";message.style.color="#DC2626";return;}message.textContent="Form validated. Opening dashboard...";message.style.color="#0F766E";setTimeout(()=>openDashboard(name),300);});
document.getElementById("demoLogin")?.addEventListener("click",()=>openDashboard("Demo User"));
