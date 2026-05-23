App({
  onLaunch() {
    // Check if user is logged in
    const token = wx.getStorageSync('token');
    if (token) {
      this.globalData.token = token;
      this.globalData.isLoggedIn = true;
    }
  },

  onShow() {
    // Handle app show
  },

  globalData: {
    token: '',
    userId: '',
    userInfo: {},
    isLoggedIn: false,
    apiBaseUrl: 'http://localhost:8080/api'  // Change to your actual domain
  },

  // API request wrapper
  request(options) {
    const { url, method = 'GET', data = {}, success, fail, complete } = options;
    const token = this.globalData.token;

    const requestUrl = this.globalData.apiBaseUrl + url;
    const headers = {
      'Content-Type': 'application/json'
    };

    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    return wx.request({
      url: requestUrl,
      method,
      data,
      header: headers,
      success: (res) => {
        if (res.statusCode === 200 || res.statusCode === 201) {
          success && success(res.data);
        } else if (res.statusCode === 401) {
          // Token expired or invalid
          this.globalData.token = '';
          this.globalData.isLoggedIn = false;
          wx.removeStorageSync('token');
          wx.navigateTo({ url: '/pages/login/login' });
          fail && fail(res);
        } else {
          fail && fail(res.data);
        }
      },
      fail,
      complete
    });
  }
});
