// pages/index/index.js
const app = getApp();

Page({
  data: {
    musics: [],
    favorites: {},
    searchKeyword: '',
    activeTab: 'all',
    currentPage: 1,
    pageSize: 20,
    loading: false,
    hasMore: true
  },

  onLoad() {
    this.checkAuth();
    this.loadMusics();
  },

  onShow() {
    // Refresh favorites when page is shown
    this.loadFavorites();
  },

  checkAuth() {
    if (!app.globalData.isLoggedIn) {
      wx.redirectTo({ url: '/pages/login/login' });
    }
  },

  loadMusics(reset = true) {
    if (this.data.loading) return;

    if (reset) {
      this.setData({ musics: [], currentPage: 1, hasMore: true });
    }

    this.setData({ loading: true });

    const url = this.data.searchKeyword
      ? `/music/search?keyword=${this.data.searchKeyword}&page=${this.data.currentPage}&size=${this.data.pageSize}`
      : `/music/list?page=${this.data.currentPage}&size=${this.data.pageSize}`;

    app.request({
      url,
      method: 'GET',
      success: (res) => {
        if (res.code === 0) {
          const newMusics = res.data.content || [];
          if (newMusics.length < this.data.pageSize) {
            this.setData({ hasMore: false });
          }

          this.setData({
            musics: reset ? newMusics : [...this.data.musics, ...newMusics],
            currentPage: this.data.currentPage + 1
          });
        }
      },
      fail: () => {
        wx.showToast({ title: 'Failed to load music', icon: 'none' });
      },
      complete: () => {
        this.setData({ loading: false });
        wx.stopPullDownRefresh();
      }
    });
  },

  loadFavorites() {
    app.request({
      url: '/favorites',
      method: 'GET',
      success: (res) => {
        if (res.code === 0) {
          const favorites = {};
          res.data.forEach(fav => {
            favorites[fav.musicId] = true;
          });
          this.setData({ favorites });
        }
      }
    });
  },

  handleSearch(e) {
    this.setData({ searchKeyword: e.detail.value });
  },

  performSearch() {
    this.loadMusics(true);
  },

  switchTab(e) {
    const tab = e.currentTarget.dataset.tab;
    this.setData({ activeTab: tab });

    if (tab === 'favorites') {
      this.loadFavorites();
    }
  },

  playMusic(e) {
    const musicId = e.currentTarget.dataset.id;
    wx.navigateTo({
      url: `/pages/player/player?id=${musicId}`
    });
  },

  toggleFavorite(e) {
    const musicId = e.currentTarget.dataset.id;
    const isFavorited = this.data.favorites[musicId];

    const method = isFavorited ? 'DELETE' : 'POST';
    const url = `/favorites/${musicId}`;

    app.request({
      url,
      method,
      success: (res) => {
        if (res.code === 0) {
          const favorites = this.data.favorites;
          if (isFavorited) {
            delete favorites[musicId];
          } else {
            favorites[musicId] = true;
          }
          this.setData({ favorites });

          wx.showToast({
            title: isFavorited ? 'Removed from favorites' : 'Added to favorites',
            icon: 'success',
            duration: 1500
          });
        }
      }
    });
  },

  loadMore() {
    if (this.data.hasMore && !this.data.loading) {
      this.loadMusics(false);
    }
  },

  onPullDownRefresh() {
    this.loadMusics(true);
  }
});
