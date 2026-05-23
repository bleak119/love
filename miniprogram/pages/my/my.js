// pages/my/my.js
const app = getApp();

Page({
  data: {
    userInfo: {}
  },

  onLoad() {
    this.checkAuth();
    this.loadUserInfo();
  },

  onShow() {
    this.loadUserInfo();
  },

  checkAuth() {
    if (!app.globalData.isLoggedIn) {
      wx.redirectTo({ url: '/pages/login/login' });
    }
  },

  loadUserInfo() {
    app.request({
      url: '/auth/me',
      method: 'GET',
      success: (res) => {
        if (res.code === 0) {
          this.setData({ userInfo: res.data });
        }
      }
    });
  },

  navigateTo(e) {
    const page = e.currentTarget.dataset.page;
    const pageMap = {
      favorites: '/pages/favorites/favorites',
      history: '/pages/history/history',
      playlists: '/pages/playlists/playlists',
      settings: '/pages/settings/settings'
    };

    if (pageMap[page]) {
      wx.navigateTo({ url: pageMap[page] });
    }
  },

  logout() {
    wx.showModal({
      title: 'Logout',
      content: 'Are you sure you want to logout?',
      success: (res) => {
        if (res.confirm) {
          // Clear local data
          app.globalData.token = '';
          app.globalData.isLoggedIn = false;
          wx.removeStorageSync('token');
          wx.removeStorageSync('userId');

          wx.showToast({
            title: 'Logged out',
            icon: 'success'
          });

          // Redirect to login
          setTimeout(() => {
            wx.redirectTo({ url: '/pages/login/login' });
          }, 1000);
        }
      }
    });
  }
});
