// pages/login/login.js
const app = getApp();

Page({
  data: {
    loading: false
  },

  onLoad() {
    // Check if already logged in
    if (app.globalData.isLoggedIn) {
      wx.redirectTo({ url: '/pages/index/index' });
    }
  },

  handleGetPhoneNumber(e) {
    if (e.detail.errMsg === 'getPhoneNumber:ok') {
      this.performLogin();
    }
  },

  performLogin() {
    this.setData({ loading: true });

    wx.login({
      success: (res) => {
        if (res.code) {
          const code = res.code;
          app.request({
            url: '/auth/wx-login',
            method: 'POST',
            data: { code },
            success: (res) => {
              if (res.code === 0) {
                // Save token and user info
                app.globalData.token = res.data.token;
                app.globalData.userId = res.data.userId;
                app.globalData.isLoggedIn = true;
                wx.setStorageSync('token', res.data.token);
                wx.setStorageSync('userId', res.data.userId);

                wx.showToast({
                  title: 'Login successful',
                  icon: 'success'
                });

                // Redirect to home
                wx.redirectTo({ url: '/pages/index/index' });
              } else {
                wx.showToast({
                  title: res.message || 'Login failed',
                  icon: 'none'
                });
              }
            },
            fail: (err) => {
              wx.showToast({
                title: 'Network error',
                icon: 'none'
              });
            },
            complete: () => {
              this.setData({ loading: false });
            }
          });
        } else {
          wx.showToast({
            title: 'Login failed',
            icon: 'none'
          });
          this.setData({ loading: false });
        }
      }
    });
  }
});
