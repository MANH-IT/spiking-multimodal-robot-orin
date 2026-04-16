// team.js - Xử lý trang đội ngũ
// Robot EEEC - Đại học Giao thông Vận tải

// ==================== DỮ LIỆU ĐỘI NGŨ ====================
const teamData = {
    // Giảng viên hướng dẫn
    advisors: [
        {
            id: 1,
            name: "PGS. TS. Trịnh Lương Miên",
            role: "Giảng viên hướng dẫn chính",
            department: "Khoa Công nghệ thông tin",
            email: "trinhlm@utc.edu.vn",
            phone: "0983.xxx.xxx",
            education: "Tiến sĩ CNTT - Đại học Bách khoa Hà Nội",
            expertise: ["Spiking Neural Networks", "Deep Learning", "Robot Học", "Computer Vision"],
            avatar: "/assets/images/team/advisor.jpg",
            initial: "TS",
            badge: "Hướng dẫn chính"
        }
    ],
    // Thành viên nhóm
    members: [
        {
            id: 2,
            name: "Phan Đình Mạnh",
            role: "Trưởng nhóm phát triển",
            department: "Kỹ thuật Robot và Trí tuệ nhân tạo K64",
            studentId: "642101",
            email: "phandinhmanh@utc.edu.vn",
            phone: "0985.xxx.xxx",
            responsibilities: "NLP System, RAG, SpikingLanguageModel, Backend API",
            skills: ["Python", "PyTorch", "snntorch", "FastAPI", "LangChain"],
            avatar: "/assets/images/team/leader.jpg",
            initial: "PM",
            badge: "Nhóm trưởng"
        },
        {
            id: 3,
            name: "Đoàn Văn Hoạt",
            role: "Thành viên nhóm phát triển",
            department: "Kỹ thuật Robot và Trí tuệ nhân tạo K64",
            studentId: "642102",
            email: "doanvanhoat@utc.edu.vn",
            phone: "0986.xxx.xxx",
            responsibilities: "Vision System, DepthAwareSNN, Web UI, Frontend",
            skills: ["OpenCV", "CUDA", "JavaScript", "HTML/CSS", "React"],
            avatar: "/assets/images/team/member.jpg",
            initial: "ĐH",
            badge: "Thành viên"
        }
    ]
};

// ==================== CẤU HÌNH ====================
const TEAM_CONFIG = {
    API_URL: 'http://localhost:8000',
    ANIMATION_DELAY: 100
};

// ==================== DOM ELEMENTS ====================
let dom = {
    advisorsGrid: null,
    membersGrid: null,
    teamCards: null
};

// ==================== UTILITY FUNCTIONS ====================
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function getInitials(name) {
    const parts = name.split(' ');
    if (parts.length >= 2) {
        return (parts[0].charAt(0) + parts[parts.length - 1].charAt(0)).toUpperCase();
    }
    return name.substring(0, 2).toUpperCase();
}

// ==================== RENDER TEAM MEMBERS ====================
function renderAdvisors() {
    if (!dom.advisorsGrid) return;

    dom.advisorsGrid.innerHTML = teamData.advisors.map(advisor => `
        <div class="team-card advisor-card" data-id="${advisor.id}">
            <div class="team-avatar">
                <img class="avatar-img" src="${advisor.avatar}" 
                     alt="${advisor.name}"
                     onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                <div class="avatar-placeholder" style="display: none;">${advisor.initial}</div>
                <div class="avatar-badge">${advisor.badge}</div>
            </div>
            <div class="team-info">
                <h3>${escapeHtml(advisor.name)}</h3>
                <div class="team-role">${escapeHtml(advisor.role)}</div>
                <div class="team-department">${escapeHtml(advisor.department)}</div>
                <div class="team-detail"><strong>Học vị:</strong> ${escapeHtml(advisor.education)}</div>
                <div class="team-detail"><strong>Email:</strong> ${escapeHtml(advisor.email)}</div>
                <div class="team-detail"><strong>Điện thoại:</strong> ${escapeHtml(advisor.phone)}</div>
                <div class="team-expertise">
                    ${advisor.expertise.map(skill => `<span class="expertise-tag">${escapeHtml(skill)}</span>`).join('')}
                </div>
            </div>
        </div>
    `).join('');
}

function renderMembers() {
    if (!dom.membersGrid) return;

    dom.membersGrid.innerHTML = teamData.members.map(member => `
        <div class="team-card" data-id="${member.id}">
            <div class="team-avatar">
                <img class="avatar-img" src="${member.avatar}" 
                     alt="${member.name}"
                     onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                <div class="avatar-placeholder" style="display: none;">${member.initial}</div>
                <div class="avatar-badge">${member.badge}</div>
            </div>
            <div class="team-info">
                <h3>${escapeHtml(member.name)}</h3>
                <div class="team-role">${escapeHtml(member.role)}</div>
                <div class="team-department">${escapeHtml(member.department)}</div>
                <div class="team-detail"><strong>MSSV:</strong> ${escapeHtml(member.studentId)}</div>
                <div class="team-detail"><strong>Email:</strong> ${escapeHtml(member.email)}</div>
                <div class="team-detail"><strong>Điện thoại:</strong> ${escapeHtml(member.phone)}</div>
                <div class="team-detail"><strong>Phụ trách:</strong> ${escapeHtml(member.responsibilities)}</div>
                <div class="team-skills">
                    ${member.skills.map(skill => `<span class="skill-tag">${escapeHtml(skill)}</span>`).join('')}
                </div>
            </div>
        </div>
    `).join('');
}

function renderAllTeams() {
    renderAdvisors();
    renderMembers();
}

// ==================== ANIMATION ON SCROLL ====================
function setupScrollAnimation() {
    const cards = document.querySelectorAll('.team-card');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }, index * 100);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    cards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = 'all 0.5s ease';
        observer.observe(card);
    });
}

// ==================== MODAL DETAIL (tùy chọn) ====================
function showTeamDetail(id) {
    const allMembers = [...teamData.advisors, ...teamData.members];
    const member = allMembers.find(m => m.id === id);
    if (!member) return;

    let modal = document.getElementById('teamModal');
    if (!modal) {
        modal = createTeamModal();
    }

    const isAdvisor = member.expertise !== undefined;

    modal.querySelector('.modal-header h3').textContent = member.name;
    modal.querySelector('.modal-role').textContent = member.role;
    modal.querySelector('.modal-department').textContent = member.department;
    modal.querySelector('.modal-email').innerHTML = `<strong>Email:</strong> ${member.email}`;
    modal.querySelector('.modal-phone').innerHTML = `<strong>Điện thoại:</strong> ${member.phone}`;

    if (isAdvisor) {
        modal.querySelector('.modal-extra').innerHTML = `<strong>Học vị:</strong> ${member.education}`;
        modal.querySelector('.modal-skills').innerHTML = `
            <strong>Chuyên môn:</strong><br>
            <div class="expertise-list">
                ${member.expertise.map(skill => `<span class="expertise-tag">${skill}</span>`).join('')}
            </div>
        `;
    } else {
        modal.querySelector('.modal-extra').innerHTML = `
            <strong>MSSV:</strong> ${member.studentId}<br>
            <strong>Phụ trách:</strong> ${member.responsibilities}
        `;
        modal.querySelector('.modal-skills').innerHTML = `
            <strong>Kỹ năng:</strong><br>
            <div class="skills-list">
                ${member.skills.map(skill => `<span class="skill-tag">${skill}</span>`).join('')}
            </div>
        `;
    }

    modal.classList.add('active');
}

function createTeamModal() {
    const modal = document.createElement('div');
    modal.id = 'teamModal';
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3></h3>
                <button class="modal-close">&times;</button>
            </div>
            <div class="modal-body">
                <div class="modal-role"></div>
                <div class="modal-department"></div>
                <div class="modal-email"></div>
                <div class="modal-phone"></div>
                <div class="modal-extra"></div>
                <div class="modal-skills"></div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);

    // Thêm styles cho modal
    const modalStyle = document.createElement('style');
    modalStyle.textContent = `
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(5px);
            z-index: 2000;
            justify-content: center;
            align-items: center;
        }
        .modal.active {
            display: flex;
        }
        .modal-content {
            background: #1a1f2e;
            border-radius: 24px;
            max-width: 500px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            border: 1px solid rgba(0, 153, 255, 0.3);
            animation: modalIn 0.3s ease;
        }
        @keyframes modalIn {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }
        .modal-header {
            padding: 1.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .modal-header h3 {
            font-size: 1.3rem;
            font-weight: 700;
            color: #ffdd44;
        }
        .modal-close {
            background: none;
            border: none;
            color: white;
            font-size: 1.8rem;
            cursor: pointer;
            transition: all 0.3s;
        }
        .modal-close:hover {
            color: #ffdd44;
            transform: scale(1.1);
        }
        .modal-body {
            padding: 1.5rem;
        }
        .modal-role {
            color: #ffdd44;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .modal-department {
            color: #8a92a8;
            font-size: 0.85rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .modal-email, .modal-phone, .modal-extra {
            font-size: 0.85rem;
            color: #b0b8cc;
            margin-bottom: 0.5rem;
        }
        .modal-skills {
            margin-top: 1rem;
            padding-top: 0.5rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        .expertise-list, .skills-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }
        .expertise-tag, .skill-tag {
            background: rgba(0, 153, 255, 0.15);
            border: 1px solid rgba(0, 153, 255, 0.2);
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.7rem;
            color: #0099ff;
        }
    `;
    document.head.appendChild(modalStyle);

    const closeBtn = modal.querySelector('.modal-close');
    closeBtn.onclick = () => modal.classList.remove('active');
    modal.onclick = (e) => {
        if (e.target === modal) modal.classList.remove('active');
    };

    return modal;
}

// ==================== TECH STACK ANIMATION ====================
function setupTechStackAnimation() {
    const techItems = document.querySelectorAll('.tech-item');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'scale(1)';
                }, index * 50);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    techItems.forEach(item => {
        item.style.opacity = '0';
        item.style.transform = 'scale(0.9)';
        item.style.transition = 'all 0.3s ease';
        observer.observe(item);
    });
}

// ==================== HOVER EFFECT FOR CARDS ====================
function setupCardHoverEffect() {
    const cards = document.querySelectorAll('.team-card');

    cards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-8px)';
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0)';
        });
    });
}

// ==================== COUNTER ANIMATION ====================
function animateTeamStats() {
    const stats = {
        advisors: teamData.advisors.length,
        members: teamData.members.length,
        total: teamData.advisors.length + teamData.members.length
    };

    // Thêm stats bar nếu cần
    const statsContainer = document.querySelector('.team-stats');
    if (statsContainer) {
        statsContainer.innerHTML = `
            <div class="team-stat-item">
                <div class="team-stat-number">${stats.advisors}</div>
                <div class="team-stat-label">Giảng viên hướng dẫn</div>
            </div>
            <div class="team-stat-item">
                <div class="team-stat-number">${stats.members}</div>
                <div class="team-stat-label">Sinh viên thực hiện</div>
            </div>
            <div class="team-stat-item">
                <div class="team-stat-number">${stats.total}</div>
                <div class="team-stat-label">Thành viên</div>
            </div>
        `;
    }
}

// ==================== IMAGE ERROR HANDLING ====================
function setupImageErrorHandling() {
    const images = document.querySelectorAll('.avatar-img');
    images.forEach(img => {
        img.addEventListener('error', function () {
            this.style.display = 'none';
            const placeholder = this.nextElementSibling;
            if (placeholder && placeholder.classList.contains('avatar-placeholder')) {
                placeholder.style.display = 'flex';
            }
        });
    });
}

// ==================== INITIALIZATION ====================
function init() {
    // Lấy DOM elements
    dom.advisorsGrid = document.getElementById('advisorsGrid');
    dom.membersGrid = document.getElementById('membersGrid');

    // Render nội dung
    renderAllTeams();

    // Thiết lập các hiệu ứng
    setupScrollAnimation();
    setupTechStackAnimation();
    setupCardHoverEffect();
    setupImageErrorHandling();
    animateTeamStats();

    // Log thông tin
    console.log('Team page initialized - Robot EEEC');
    console.log(`Advisors: ${teamData.advisors.length}`);
    console.log(`Members: ${teamData.members.length}`);

    // Thêm styles cho skill tags nếu chưa có
    if (!document.querySelector('#teamStyles')) {
        const style = document.createElement('style');
        style.id = 'teamStyles';
        style.textContent = `
            .team-skills {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-top: 1rem;
            }
            .skill-tag {
                background: rgba(255, 221, 68, 0.12);
                border: 1px solid rgba(255, 221, 68, 0.2);
                padding: 0.25rem 0.7rem;
                border-radius: 20px;
                font-size: 0.7rem;
                color: #ffdd44;
            }
            .team-stat-item {
                text-align: center;
                padding: 1rem;
            }
            .team-stat-number {
                font-size: 2rem;
                font-weight: 800;
                background: linear-gradient(135deg, #ffdd44, #ffaa00);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
            }
            .team-stat-label {
                font-size: 0.8rem;
                color: #8a92a8;
            }
        `;
        document.head.appendChild(style);
    }
}

// Khởi tạo khi DOM ready
document.addEventListener('DOMContentLoaded', init);