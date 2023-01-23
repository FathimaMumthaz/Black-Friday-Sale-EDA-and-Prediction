{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "Problem Statement\n",
        "\n",
        "A retail company wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month. The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.\n",
        "\n",
        "Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products."
      ],
      "metadata": {
        "id": "cyUeyzWAEAx5"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5Z4_RVit8coU"
      },
      "outputs": [],
      "source": [
        "# User_ID\tUser ID\n",
        "# Product_ID\tProduct ID\n",
        "# Gender\tSex of User\n",
        "# Age\tAge in bins\n",
        "# Occupation\tOccupation (Masked)\n",
        "# City_Category\tCategory of the City (A,B,C)\n",
        "# Stay_In_Current_City_Years\tNumber of years stay in current city\n",
        "# Marital_Status\tMarital Status\n",
        "# Product_Category_1\tProduct Category (Masked)\n",
        "# Product_Category_2\tProduct may belongs to other category also (Masked)\n",
        "# Product_Category_3\tProduct may belongs to other category also (Masked)\n",
        "# Purchase\tPurchase Amount (Target Variable)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns"
      ],
      "metadata": {
        "id": "g51Gez-t8ihP"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data = pd.read_csv(\"https://raw.githubusercontent.com/nanthasnk/Black-Friday-Sales-Prediction/master/Data/BlackFridaySales.csv\")"
      ],
      "metadata": {
        "id": "8_s2JluB8nCR"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 339
        },
        "id": "aAWWrgjj8sXj",
        "outputId": "b744f6e6-fd96-4d0e-9cde-d93bceee34a6"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   User_ID Product_ID Gender   Age  Occupation City_Category  \\\n",
              "0  1000001  P00069042      F  0-17          10             A   \n",
              "1  1000001  P00248942      F  0-17          10             A   \n",
              "2  1000001  P00087842      F  0-17          10             A   \n",
              "3  1000001  P00085442      F  0-17          10             A   \n",
              "4  1000002  P00285442      M   55+          16             C   \n",
              "\n",
              "  Stay_In_Current_City_Years  Marital_Status  Product_Category_1  \\\n",
              "0                          2               0                   3   \n",
              "1                          2               0                   1   \n",
              "2                          2               0                  12   \n",
              "3                          2               0                  12   \n",
              "4                         4+               0                   8   \n",
              "\n",
              "   Product_Category_2  Product_Category_3  Purchase  \n",
              "0                 NaN                 NaN      8370  \n",
              "1                 6.0                14.0     15200  \n",
              "2                 NaN                 NaN      1422  \n",
              "3                14.0                 NaN      1057  \n",
              "4                 NaN                 NaN      7969  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-573926e6-05ac-4b8b-af1d-795f1b1ae42b\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>User_ID</th>\n",
              "      <th>Product_ID</th>\n",
              "      <th>Gender</th>\n",
              "      <th>Age</th>\n",
              "      <th>Occupation</th>\n",
              "      <th>City_Category</th>\n",
              "      <th>Stay_In_Current_City_Years</th>\n",
              "      <th>Marital_Status</th>\n",
              "      <th>Product_Category_1</th>\n",
              "      <th>Product_Category_2</th>\n",
              "      <th>Product_Category_3</th>\n",
              "      <th>Purchase</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1000001</td>\n",
              "      <td>P00069042</td>\n",
              "      <td>F</td>\n",
              "      <td>0-17</td>\n",
              "      <td>10</td>\n",
              "      <td>A</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>8370</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1000001</td>\n",
              "      <td>P00248942</td>\n",
              "      <td>F</td>\n",
              "      <td>0-17</td>\n",
              "      <td>10</td>\n",
              "      <td>A</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>6.0</td>\n",
              "      <td>14.0</td>\n",
              "      <td>15200</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1000001</td>\n",
              "      <td>P00087842</td>\n",
              "      <td>F</td>\n",
              "      <td>0-17</td>\n",
              "      <td>10</td>\n",
              "      <td>A</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>12</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1422</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1000001</td>\n",
              "      <td>P00085442</td>\n",
              "      <td>F</td>\n",
              "      <td>0-17</td>\n",
              "      <td>10</td>\n",
              "      <td>A</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>12</td>\n",
              "      <td>14.0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1057</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>1000002</td>\n",
              "      <td>P00285442</td>\n",
              "      <td>M</td>\n",
              "      <td>55+</td>\n",
              "      <td>16</td>\n",
              "      <td>C</td>\n",
              "      <td>4+</td>\n",
              "      <td>0</td>\n",
              "      <td>8</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>7969</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-573926e6-05ac-4b8b-af1d-795f1b1ae42b')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-573926e6-05ac-4b8b-af1d-795f1b1ae42b button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-573926e6-05ac-4b8b-af1d-795f1b1ae42b');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "I7hZE1S68wyF",
        "outputId": "f527603f-58bf-4049-f4a2-808bd8287a84"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(550068, 12)"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9m2fT9MZ837b",
        "outputId": "2dd4c579-74bc-4f93-f1d7-b40630ffc158"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 550068 entries, 0 to 550067\n",
            "Data columns (total 12 columns):\n",
            " #   Column                      Non-Null Count   Dtype  \n",
            "---  ------                      --------------   -----  \n",
            " 0   User_ID                     550068 non-null  int64  \n",
            " 1   Product_ID                  550068 non-null  object \n",
            " 2   Gender                      550068 non-null  object \n",
            " 3   Age                         550068 non-null  object \n",
            " 4   Occupation                  550068 non-null  int64  \n",
            " 5   City_Category               550068 non-null  object \n",
            " 6   Stay_In_Current_City_Years  550068 non-null  object \n",
            " 7   Marital_Status              550068 non-null  int64  \n",
            " 8   Product_Category_1          550068 non-null  int64  \n",
            " 9   Product_Category_2          376430 non-null  float64\n",
            " 10  Product_Category_3          166821 non-null  float64\n",
            " 11  Purchase                    550068 non-null  int64  \n",
            "dtypes: float64(2), int64(5), object(5)\n",
            "memory usage: 50.4+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "h-6ZLLRE88ey",
        "outputId": "b4be3a01-30b6-4cc0-834c-1ba6f027f677"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "User_ID                            0\n",
              "Product_ID                         0\n",
              "Gender                             0\n",
              "Age                                0\n",
              "Occupation                         0\n",
              "City_Category                      0\n",
              "Stay_In_Current_City_Years         0\n",
              "Marital_Status                     0\n",
              "Product_Category_1                 0\n",
              "Product_Category_2            173638\n",
              "Product_Category_3            383247\n",
              "Purchase                           0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.isnull().sum()/data.shape[0]*100"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sWQTzCnA8-iG",
        "outputId": "933ff663-4328-4079-d449-f3f5bee4fbce"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "User_ID                        0.000000\n",
              "Product_ID                     0.000000\n",
              "Gender                         0.000000\n",
              "Age                            0.000000\n",
              "Occupation                     0.000000\n",
              "City_Category                  0.000000\n",
              "Stay_In_Current_City_Years     0.000000\n",
              "Marital_Status                 0.000000\n",
              "Product_Category_1             0.000000\n",
              "Product_Category_2            31.566643\n",
              "Product_Category_3            69.672659\n",
              "Purchase                       0.000000\n",
              "dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.nunique()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ClPEk11x9JW7",
        "outputId": "35c031a1-e62e-41f1-f364-4e4201aace41"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "User_ID                        5891\n",
              "Product_ID                     3631\n",
              "Gender                            2\n",
              "Age                               7\n",
              "Occupation                       21\n",
              "City_Category                     3\n",
              "Stay_In_Current_City_Years        5\n",
              "Marital_Status                    2\n",
              "Product_Category_1               20\n",
              "Product_Category_2               17\n",
              "Product_Category_3               15\n",
              "Purchase                      18105\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.distplot(data[\"Purchase\"],color='r')\n",
        "plt.title(\"Purchase Distribution\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 350
        },
        "id": "rFrdn-v09MwA",
        "outputId": "c7330f97-1138-42d4-ca22-a94124b6411a"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).\n",
            "  warnings.warn(msg, FutureWarning)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.boxplot(data[\"Purchase\"])\n",
        "plt.title(\"Boxplot of Purchase\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 350
        },
        "id": "2222ppmF9d2I",
        "outputId": "55b71ef8-2a8c-4bfc-ad57-e3406c37992a"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data[\"Purchase\"].skew()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "US2glLBC9ni_",
        "outputId": "03324957-9d29-4da4-9965-b8bbf2b9166a"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.6001400037087128"
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data[\"Purchase\"].kurtosis()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "D3kWlIyi9pxN",
        "outputId": "2e7da31c-715a-4559-d68b-15104d3a84d6"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "-0.3383775655851702"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data[\"Purchase\"].describe()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CJoaLhBP9sIm",
        "outputId": "a921a8ae-3a73-42be-a8cf-36d9231fdfbc"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "count    550068.000000\n",
              "mean       9263.968713\n",
              "std        5023.065394\n",
              "min          12.000000\n",
              "25%        5823.000000\n",
              "50%        8047.000000\n",
              "75%       12054.000000\n",
              "max       23961.000000\n",
              "Name: Purchase, dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.countplot(data['Gender'])\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 334
        },
        "id": "P-FdBo8699PW",
        "outputId": "60a3efb5-d961-4fb7-a50c-63df80ceb9bd"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZcAAAEGCAYAAACpXNjrAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAYWklEQVR4nO3df7BfdX3n8efLBITa8kO4ZWlCN6xmpxNZDZICu3ZmWRgh0F2DFp0w25LajHErODrtdoHuzKIoMzqtpeIiu1giwVoDi7Vk3dg0C7hOO+XHRVMgIMNdkJIsktskgNYCA773j+8n9Uv43psLnO/3ws3zMXPmnvM+n885n+9M5OU55/P9nlQVkiR16XWzPQBJ0txjuEiSOme4SJI6Z7hIkjpnuEiSOjd/tgfwanHkkUfWokWLZnsYkvSactddd/1dVY3tXTdcmkWLFjE+Pj7bw5Ck15Qkjwyqe1tMktQ5w0WS1DnDRZLUOcNFktQ5w0WS1DnDRZLUOcNFktQ5w0WS1DnDRZLUOb+hL+0H/vbSfzHbQ9Cr0M//l3uGdmyvXCRJnTNcJEmdM1wkSZ0bergkmZfkO0m+3raPTXJ7kokk1yc5sNVf37Yn2v5Ffce4uNUfSHJGX315q00kuaivPvAckqTRGMWVy0eA+/u2Pw1cXlVvBnYDq1t9NbC71S9v7UiyBFgJvAVYDny+BdY84ErgTGAJcG5rO905JEkjMNRwSbIQ+GXgj9p2gFOBG1uTdcDZbX1F26btP621XwGsr6pnquphYAI4sS0TVfVQVT0LrAdW7OMckqQRGPaVyx8C/wn4cds+Aniiqp5r29uABW19AfAoQNv/ZGv/j/W9+kxVn+4cL5BkTZLxJOOTk5Mv9zNKkvYytHBJ8m+BHVV117DO8UpV1dVVtayqlo2NvegtnZKkl2mYX6J8B/CuJGcBBwGHAJ8FDksyv11ZLAS2t/bbgWOAbUnmA4cCO/vqe/T3GVTfOc05JEkjMLQrl6q6uKoWVtUieg/kb6mqfw/cCpzTmq0CbmrrG9o2bf8tVVWtvrLNJjsWWAzcAdwJLG4zww5s59jQ+kx1DknSCMzG91wuBH4ryQS95yPXtPo1wBGt/lvARQBVtRW4AbgP+HPg/Kp6vl2VXABsojcb7YbWdrpzSJJGYCS/LVZV3wS+2dYfojfTa+82TwPvnaL/ZcBlA+obgY0D6gPPIUkaDb+hL0nqnOEiSeqc4SJJ6pzhIknqnOEiSeqc4SJJ6pzhIknqnOEiSeqc4SJJ6pzhIknqnOEiSeqc4SJJ6pzhIknqnOEiSeqc4SJJ6pzhIknq3NDCJclBSe5I8jdJtib5eKtfm+ThJFvasrTVk+SKJBNJ7k7y9r5jrUryYFtW9dVPSHJP63NFkrT6G5Nsbu03Jzl8WJ9TkvRiw7xyeQY4tareBiwFlic5ue37napa2pYtrXYmsLgta4CroBcUwCXASfTeLnlJX1hcBXygr9/yVr8IuLmqFgM3t21J0ogMLVyq54dt84C21DRdVgDXtX63AYclORo4A9hcVbuqajewmV5QHQ0cUlW3VVUB1wFn9x1rXVtf11eXJI3AUJ+5JJmXZAuwg15A3N52XdZufV2e5PWttgB4tK/7tlabrr5tQB3gqKp6rK1/HzhqivGtSTKeZHxycvLlfUhJ0osMNVyq6vmqWgosBE5MchxwMfALwC8CbwQuHPIYiimumKrq6qpaVlXLxsbGhjkMSdqvjGS2WFU9AdwKLK+qx9qtr2eAL9J7jgKwHTimr9vCVpuuvnBAHeDxdtuM9ndHt59IkjSdYc4WG0tyWFs/GHgn8N2+/+iH3rOQe1uXDcB5bdbYycCT7dbWJuD0JIe3B/mnA5vavqeSnNyOdR5wU9+x9swqW9VXlySNwPwhHvtoYF2SefRC7Iaq+nqSW5KMAQG2AP+htd8InAVMAD8C3g9QVbuSfAK4s7W7tKp2tfUPAdcCBwPfaAvAp4AbkqwGHgHeN7RPKUl6kaGFS1XdDRw/oH7qFO0LOH+KfWuBtQPq48BxA+o7gdNe4pAlSR3xG/qSpM4ZLpKkzhkukqTOGS6SpM4ZLpKkzhkukqTOGS6SpM4ZLpKkzhkukqTOGS6SpM4ZLpKkzhkukqTOGS6SpM4ZLpKkzhkukqTOGS6SpM4N8zXHByW5I8nfJNma5OOtfmyS25NMJLk+yYGt/vq2PdH2L+o71sWt/kCSM/rqy1ttIslFffWB55AkjcYwr1yeAU6tqrcBS4HlSU4GPg1cXlVvBnYDq1v71cDuVr+8tSPJEmAl8BZgOfD5JPPa65OvBM4ElgDntrZMcw5J0ggMLVyq54dt84C2FHAqcGOrrwPObusr2jZt/2lJ0urrq+qZqnoYmABObMtEVT1UVc8C64EVrc9U55AkjcBQn7m0K4wtwA5gM/B/gSeq6rnWZBuwoK0vAB4FaPufBI7or+/VZ6r6EdOcY+/xrUkynmR8cnLylXxUSVKfoYZLVT1fVUuBhfSuNH5hmOd7qarq6qpaVlXLxsbGZns4kjRnjGS2WFU9AdwK/EvgsCTz266FwPa2vh04BqDtPxTY2V/fq89U9Z3TnEOSNALDnC02luSwtn4w8E7gfnohc05rtgq4qa1vaNu0/bdUVbX6yjab7FhgMXAHcCewuM0MO5DeQ/8Nrc9U55AkjcD8fTd52Y4G1rVZXa8Dbqiqrye5D1if5JPAd4BrWvtrgC8lmQB20QsLqmprkhuA+4DngPOr6nmAJBcAm4B5wNqq2tqOdeEU55AkjcDQwqWq7gaOH1B/iN7zl73rTwPvneJYlwGXDahvBDbO9BySpNHwG/qSpM4ZLpKkzhkukqTOGS6SpM4ZLpKkzhkukqTOGS6SpM4ZLpKkzhkukqTOGS6SpM4ZLpKkzhkukqTOGS6SpM4ZLpKkzhkukqTOGS6SpM4N8zXHxyS5Ncl9SbYm+UirfyzJ9iRb2nJWX5+Lk0wkeSDJGX315a02keSivvqxSW5v9evb645pr0S+vtVvT7JoWJ9TkvRiw7xyeQ747apaApwMnJ9kSdt3eVUtbctGgLZvJfAWYDnw+STz2muSrwTOBJYA5/Yd59PtWG8GdgOrW301sLvVL2/tJEkjMrRwqarHqurbbf0HwP3Agmm6rADWV9UzVfUwMEHvVcUnAhNV9VBVPQusB1YkCXAqcGPrvw44u+9Y69r6jcBprb0kaQRG8syl3ZY6Hri9lS5IcneStUkOb7UFwKN93ba12lT1I4Anquq5veovOFbb/2Rrv/e41iQZTzI+OTn5ij6jJOknhh4uSX4a+Crw0ap6CrgKeBOwFHgM+MywxzCVqrq6qpZV1bKxsbHZGoYkzTlDDZckB9ALli9X1Z8CVNXjVfV8Vf0Y+AK9214A24Fj+rovbLWp6juBw5LM36v+gmO1/Ye29pKkERjmbLEA1wD3V9Uf9NWP7mv2buDetr4BWNlmeh0LLAbuAO4EFreZYQfSe+i/oaoKuBU4p/VfBdzUd6xVbf0c4JbWXpI0AvP33eRlewfwa8A9Sba02u/Sm+21FCjge8AHAapqa5IbgPvozTQ7v6qeB0hyAbAJmAesraqt7XgXAuuTfBL4Dr0wo/39UpIJYBe9QJIkjciMwiXJzVV12r5q/arqL4FBM7Q2TtPnMuCyAfWNg/pV1UP85LZaf/1p4L1TnUeSNFzThkuSg4CfAo5ss7r2hMUhTD+tWJK0H9vXlcsHgY8CPwfcxU/C5Sngvw5xXJKk17Bpw6WqPgt8NsmHq+pzIxqTJOk1bkbPXKrqc0n+FbCov09VXTekcUmSXsNm+kD/S/S++LgFeL6VCzBcJEkvMtOpyMuAJX5XRJI0EzP9EuW9wD8Z5kAkSXPHTK9cjgTuS3IH8MyeYlW9ayijkiS9ps00XD42zEFIkuaWmc4W+z/DHogkae6Y6WyxH9CbHQZwIHAA8PdVdciwBiZJeu2a6ZXLz+xZb792vILeq4slSXqRl/yT+9XzZ8AZQxiPJGkOmOltsff0bb6O3vdenh7KiCRJr3kznS327/rWn6P3HpYVnY9GkjQnzPSZy/uHPRBJ0twxo2cuSRYm+VqSHW35apKF++hzTJJbk9yXZGuSj7T6G5NsTvJg+3t4qyfJFUkmktyd5O19x1rV2j+YZFVf/YQk97Q+V7TJBlOeQ5I0GjN9oP9Feu+l/7m2/M9Wm85zwG9X1RJ6M8vOT7IEuAi4uaoWAze3bYAzgcVtWQNcBb2gAC4BTqL31slL+sLiKuADff2Wt/pU55AkjcBMw2Wsqr5YVc+15VpgbLoOVfVYVX27rf8AuJ/e2ytXAOtas3XA2W19BXBdm412G3BYkqPpzUrbXFW7qmo3sBlY3vYdUlW3tR/UvG6vYw06hyRpBGYaLjuT/GqSeW35VWDnTE+SZBFwPHA7cFRVPdZ2fR84qq0vAB7t67at1aarbxtQZ5pz7D2uNUnGk4xPTk7O9ONIkvZhpuHyG8D76P2H+jHgHODXZ9IxyU8DXwU+WlVP9e9rVxxD/Rn/6c5RVVdX1bKqWjY2Nu2FmCTpJZhpuFwKrKqqsar6WXph8/F9dUpyAL1g+XJV/WkrP95uadH+7mj17cAxfd0Xttp09YUD6tOdQ5I0AjMNl7e25x0AVNUuere5ptRmbl0D3F9Vf9C3awOwZ8bXKuCmvvp5bdbYycCT7dbWJuD0JIe3B/mnA5vavqeSnNzOdd5exxp0DknSCMz0S5SvS3L4noBpM7j21fcdwK8B9yTZ0mq/C3wKuCHJauARerfbADYCZwETwI+A90MvyJJ8Ariztbu0hRvAh4BrgYOBb7SFac4hSRqBmYbLZ4C/TvI/2vZ7gcum61BVfwlkit2nDWhfwPlTHGstsHZAfRw4bkB956BzSJJGY6bf0L8uyThwaiu9p6ruG96wJEmvZTO9cqGFiYEiSdqnl/yT+5Ik7YvhIknqnOEiSeqc4SJJ6pzhIknqnOEiSeqc4SJJ6pzhIknqnOEiSeqc4SJJ6pzhIknqnOEiSeqc4SJJ6pzhIknq3NDCJcnaJDuS3NtX+1iS7Um2tOWsvn0XJ5lI8kCSM/rqy1ttIslFffVjk9ze6tcnObDVX9+2J9r+RcP6jJKkwYZ55XItsHxA/fKqWtqWjQBJlgArgbe0Pp9PMi/JPOBK4ExgCXBuawvw6XasNwO7gdWtvhrY3eqXt3aSpBEaWrhU1beAXfts2LMCWF9Vz1TVw8AEcGJbJqrqoap6FlgPrEgSem/FvLH1Xwec3XesdW39RuC01l6SNCKz8czlgiR3t9tmh7faAuDRvjbbWm2q+hHAE1X13F71Fxyr7X+ytX+RJGuSjCcZn5ycfOWfTJIEjD5crgLeBCwFHgM+M+Lzv0BVXV1Vy6pq2djY2GwORZLmlJGGS1U9XlXPV9WPgS/Qu+0FsB04pq/pwlabqr4TOCzJ/L3qLzhW239oay9JGpGRhkuSo/s23w3smUm2AVjZZnodCywG7gDuBBa3mWEH0nvov6GqCrgVOKf1XwXc1HesVW39HOCW1l6SNCLz993k5UnyFeAU4Mgk24BLgFOSLAUK+B7wQYCq2prkBuA+4Dng/Kp6vh3nAmATMA9YW1Vb2ykuBNYn+STwHeCaVr8G+FKSCXoTClYO6zNKkgYbWrhU1bkDytcMqO1pfxlw2YD6RmDjgPpD/OS2Wn/9aeC9L2mwkqRO+Q19SVLnhnblsj864Xeum+0h6FXort87b7aHII2cVy6SpM4ZLpKkzhkukqTOGS6SpM4ZLpKkzhkukqTOGS6SpM4ZLpKkzhkukqTOGS6SpM4ZLpKkzhkukqTOGS6SpM4ZLpKkzg0tXJKsTbIjyb19tTcm2Zzkwfb38FZPkiuSTCS5O8nb+/qsau0fTLKqr35CkntanyuSZLpzSJJGZ5hXLtcCy/eqXQTcXFWLgZvbNsCZwOK2rAGugl5Q0Hs98kn03jp5SV9YXAV8oK/f8n2cQ5I0IkMLl6r6Fr132PdbAaxr6+uAs/vq11XPbcBhSY4GzgA2V9WuqtoNbAaWt32HVNVtVVXAdXsda9A5JEkjMupnLkdV1WNt/fvAUW19AfBoX7ttrTZdfduA+nTneJEka5KMJxmfnJx8GR9HkjTIrD3Qb1ccNZvnqKqrq2pZVS0bGxsb5lAkab8y6nB5vN3Sov3d0erbgWP62i1stenqCwfUpzuHJGlERh0uG4A9M75WATf11c9rs8ZOBp5st7Y2AacnObw9yD8d2NT2PZXk5DZL7Ly9jjXoHJKkEZk/rAMn+QpwCnBkkm30Zn19CrghyWrgEeB9rflG4CxgAvgR8H6AqtqV5BPAna3dpVW1Z5LAh+jNSDsY+EZbmOYckqQRGVq4VNW5U+w6bUDbAs6f4jhrgbUD6uPAcQPqOwedQ5I0On5DX5LUOcNFktQ5w0WS1DnDRZLUOcNFktQ5w0WS1DnDRZLUOcNFktQ5w0WS1DnDRZLUOcNFktQ5w0WS1DnDRZLUOcNFktQ5w0WS1DnDRZLUuVkJlyTfS3JPki1JxlvtjUk2J3mw/T281ZPkiiQTSe5O8va+46xq7R9MsqqvfkI7/kTrm9F/Sknaf83mlcu/qaqlVbWsbV8E3FxVi4Gb2zbAmcDitqwBroJeGNF7dfJJwInAJXsCqbX5QF+/5cP/OJKkPV5Nt8VWAOva+jrg7L76ddVzG3BYkqOBM4DNVbWrqnYDm4Hlbd8hVXVbe33ydX3HkiSNwGyFSwF/keSuJGta7aiqeqytfx84qq0vAB7t67ut1aarbxtQf5Eka5KMJxmfnJx8JZ9HktRn/iyd95eqanuSnwU2J/lu/86qqiQ17EFU1dXA1QDLli0b+vkkaX8xK1cuVbW9/d0BfI3eM5PH2y0t2t8drfl24Ji+7gtbbbr6wgF1SdKIjDxckrwhyc/sWQdOB+4FNgB7ZnytAm5q6xuA89qssZOBJ9vts03A6UkObw/yTwc2tX1PJTm5zRI7r+9YkqQRmI3bYkcBX2uzg+cDf1JVf57kTuCGJKuBR4D3tfYbgbOACeBHwPsBqmpXkk8Ad7Z2l1bVrrb+IeBa4GDgG22RJI3IyMOlqh4C3jagvhM4bUC9gPOnONZaYO2A+jhw3CserCTpZXk1TUWWJM0RhoskqXOGiySpc4aLJKlzhoskqXOGiySpc4aLJKlzhoskqXOGiySpc4aLJKlzhoskqXOGiySpc4aLJKlzhoskqXOGiySpc4aLJKlzczZckixP8kCSiSQXzfZ4JGl/MifDJck84ErgTGAJcG6SJbM7Kknaf8zJcAFOBCaq6qGqehZYD6yY5TFJ0n5j/mwPYEgWAI/2bW8DTtq7UZI1wJq2+cMkD4xgbPuLI4G/m+1BvBrk91fN9hD0Qv7b3OOSdHGUfzqoOFfDZUaq6mrg6tkex1yUZLyqls32OKS9+W9zNObqbbHtwDF92wtbTZI0AnM1XO4EFic5NsmBwEpgwyyPSZL2G3PytlhVPZfkAmATMA9YW1VbZ3lY+xtvN+rVyn+bI5Cqmu0xSJLmmLl6W0ySNIsMF0lS5wwXdSrJ80m29C2LZntMUpJK8sd92/OTTCb5+myOay6bkw/0Nav+oaqWzvYgpL38PXBckoOr6h+Ad+LXE4bKKxdJ+4uNwC+39XOBr8ziWOY8w0VdO7jvltjXZnswUp/1wMokBwFvBW6f5fHMad4WU9e8LaZXpaq6uz0DPJfeVYyGyHCRtD/ZAPw+cApwxOwOZW4zXCTtT9YCT1TVPUlOme3BzGWGi6T9RlVtA66Y7XHsD/z5F0lS55wtJknqnOEiSeqc4SJJ6pzhIknqnOEiSeqc4SINUZKjkvxJkoeS3JXkr5O8u4PjnuIv+urVzHCRhiRJgD8DvlVV/6yqTgBWAgtnYSx+p00jZbhIw3Mq8GxV/bc9hap6pKo+l2Rekt9LcmeSu5N8EP7xiuSbSW5M8t0kX24hRZLlrfZt4D17jpnkDUnWJrkjyXeSrGj1X0+yIcktwM0j/eTa7/n/ZqTheQvw7Sn2rQaerKpfTPJ64K+S/EXbd3zr+/+AvwLekWQc+AK9wJoAru871n8Gbqmq30hyGHBHkv/d9r0deGtV7eryg0n7YrhII5LkSuCXgGeBR4C3Jjmn7T4UWNz23dF+poQkW4BFwA+Bh6vqwVb/Y2BN63s68K4k/7FtHwT8fFvfbLBoNhgu0vBsBX5lz0ZVnZ/kSGAc+Fvgw1W1qb9D+zHFZ/pKz7Pv/50G+JWqemCvY51E7w2M0sj5zEUanluAg5L8Zl/tp9rfTcBvJjkAIMk/T/KGaY71XWBRkje17XP79m0CPtz3bOb4TkYvvQKGizQk1ftV2LOBf53k4SR3AOuAC4E/Au4Dvp3kXuC/M80VSlU9Te822P9qD/R39O3+BHAAcHeSrW1bmlX+KrIkqXNeuUiSOme4SJI6Z7hIkjpnuEiSOme4SJI6Z7hIkjpnuEiSOvf/Abwn4ZG6fsVxAAAAAElFTkSuQmCC\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data['Gender'].value_counts(normalize=True)*100"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gFUA1VE-9_pa",
        "outputId": "f86dc969-c898-40e9-f68a-35c9c0c2cbc5"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "M    75.310507\n",
              "F    24.689493\n",
              "Name: Gender, dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.groupby(\"Gender\").mean()[\"Purchase\"]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AcLIokhP-DDO",
        "outputId": "b59cdf02-70be-43c9-e60e-ee5d7c7f18c1"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Gender\n",
              "F    8734.565765\n",
              "M    9437.526040\n",
              "Name: Purchase, dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.countplot(data['Marital_Status'])\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 335
        },
        "id": "XPzVRYQx-Fmp",
        "outputId": "126cf84c-edce-4202-9b8a-75cfff7cd2cb"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.groupby(\"Marital_Status\").mean()[\"Purchase\"]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Z_nyyqay-SNz",
        "outputId": "b2eff5cf-313d-4887-ddc3-2700ecf66173"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Marital_Status\n",
              "0    9265.907619\n",
              "1    9261.174574\n",
              "Name: Purchase, dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.groupby(\"Marital_Status\").mean()[\"Purchase\"].plot(kind='bar')\n",
        "plt.title(\"Marital_Status and Purchase Analysis\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 293
        },
        "id": "e4L2ZjTo-UaN",
        "outputId": "0566ca06-f21a-40eb-a83c-ce848ced72b3"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(18,5))\n",
        "sns.countplot(data['Occupation'])\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 389
        },
        "id": "sE0FOvHg-Xaz",
        "outputId": "2a6188bc-c71a-4c28-af0d-59271f26fdc1"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1296x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "occup = pd.DataFrame(data.groupby(\"Occupation\").mean()[\"Purchase\"])\n",
        "occup"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 739
        },
        "id": "OxjRl1vs-iGb",
        "outputId": "b9897ecc-69b9-433d-fe4a-233c74332fb0"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "               Purchase\n",
              "Occupation             \n",
              "0           9124.428588\n",
              "1           8953.193270\n",
              "2           8952.481683\n",
              "3           9178.593088\n",
              "4           9213.980251\n",
              "5           9333.149298\n",
              "6           9256.535691\n",
              "7           9425.728223\n",
              "8           9532.592497\n",
              "9           8637.743761\n",
              "10          8959.355375\n",
              "11          9213.845848\n",
              "12          9796.640239\n",
              "13          9306.351061\n",
              "14          9500.702772\n",
              "15          9778.891163\n",
              "16          9394.464349\n",
              "17          9821.478236\n",
              "18          9169.655844\n",
              "19          8710.627231\n",
              "20          8836.494905"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-fe31af5c-fa53-4dfd-acc3-7666f973c71e\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Purchase</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Occupation</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9124.428588</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>8953.193270</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>8952.481683</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>9178.593088</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>9213.980251</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>9333.149298</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>9256.535691</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>9425.728223</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>9532.592497</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>8637.743761</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10</th>\n",
              "      <td>8959.355375</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>11</th>\n",
              "      <td>9213.845848</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>12</th>\n",
              "      <td>9796.640239</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>13</th>\n",
              "      <td>9306.351061</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>14</th>\n",
              "      <td>9500.702772</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>15</th>\n",
              "      <td>9778.891163</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>16</th>\n",
              "      <td>9394.464349</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>17</th>\n",
              "      <td>9821.478236</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>18</th>\n",
              "      <td>9169.655844</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19</th>\n",
              "      <td>8710.627231</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>20</th>\n",
              "      <td>8836.494905</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-fe31af5c-fa53-4dfd-acc3-7666f973c71e')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-fe31af5c-fa53-4dfd-acc3-7666f973c71e button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-fe31af5c-fa53-4dfd-acc3-7666f973c71e');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "occup.plot(kind='bar',figsize=(15,5))\n",
        "plt.title(\"Occupation and Purchase Analysis\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 352
        },
        "id": "NWK7NHQC-4BP",
        "outputId": "a5c29a92-0cf8-4e2b-e9cb-607aad66b84c"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1080x360 with 1 Axes>"
            ],
            "image/png": "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SVrD3XPPPcycOdOAN0lJmDlz5gr1hE72nLzHVNUNbfpG4DFtemvguoH1Fra2pbUvnKB9QkkOT3JOknMWLVo0ydIlSZIkrU4GvOGs6PM39IVXWg/cajmHrqqOrqp5VTVv1qxH3PNPkiRJkqa9yd4M/aYkW1XVDW3I5a9b+/XANgPrzW5t1wN7j2s/o7XPnmB9SZIkST00563/uVL3d+379lvmOmuttRY77bQTixcvZvvtt+eEE05g/fXXH+q473znO9lwww3527/926H2sypMtifvVODQNn0ocMpA+yHp7AXc0YZ1ngbsm2SzJJsB+wKntWW/SbJXuzLnIQP7kiRJkqShPfrRj+b888/noosuYt111+Woo45a7m3vv//+VVjZqrHMkJfkC8CPge2SLExyGPA+4DlJrgCe3eahuzrm1cCVwCeB1wNU1a3Ae4Cz29e7WxttnWPaNlfhlTUlSZIkrSLPeMYzuPLKKznjjDPYf//9H2x/4xvfyPHHHw/AnDlzeMtb3sKuu+7KySefzLe+9S123XVXdt55Z/bZZ58Ht7nkkkvYe++9ecITnsBHPvKRB9sPPPBAdtttN5761Kdy9NFHA11YfOUrX8mOO+7ITjvtxAc/+EEArrrqKubPn89uu+3GM57xDC677LKhH+Myh2tW1cuWsGif8Q3t/Lw3LGE/xwHHTdB+DrDjsuqQJEmS1lSjGKKoR1q8eDHf/OY3mT9//jLXnTlzJueddx6LFi1i11135Qc/+AFz587l1lsfuvHAZZddxve+9z1++9vfst122/G6172OddZZh+OOO47NN9+cu+++m913350XvehFXHvttVx//fVcdNFFANx+++0AHH744Rx11FFsu+22nHXWWbz+9a/nu9/97lCPc7Ln5EmSJD2Cb2T7y9dWa7K7776bXXbZBeh68g477DB+9KMfLXWbl7zkJQCceeaZPPOZz3zwHnWbb775g+vst99+zJgxgxkzZrDlllty0003MXv2bD7ykY/w1a9+FYDrrruOK664gu22246rr76aN73pTey3337su+++3HnnnfzoRz/ioIMOenCf995779CP15AnSZIkqdfGzskbtPbaa/PAAw88OD/+PnQbbLDBMvc7Y8aMB6fXWmstFi9ezBlnnMF3vvMdfvzjH7P++uuz9957c88997DZZptxwQUXcNppp3HUUUdx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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.countplot(data['City_Category'])\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 335
        },
        "id": "sUl269od-73t",
        "outputId": "b3a16e9a-a2dd-4a8d-d868-c6f142346134"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.groupby(\"City_Category\").mean()[\"Purchase\"].plot(kind='bar')\n",
        "plt.title(\"City Category and Purchase Analysis\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 293
        },
        "id": "DFEh4dYa--c9",
        "outputId": "cc001c61-d59a-4e72-b114-db761cb01906"
      },
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.countplot(data['Stay_In_Current_City_Years'])\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 335
        },
        "id": "lHG7neEr_B_4",
        "outputId": "8fc1307f-16ae-4a81-b89d-7fdd5c316d29"
      },
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZgAAAEHCAYAAACTC1DDAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAcnklEQVR4nO3df5QV5Z3n8fdH8NeMYcDYwzKAAybEs8Q4qL3KxDFx/QnmByZjXDhHQcNKPGLUM2YjTuasxsSzZrMmq4kyi5EIGRUdjSM6OMhhUHfdoDTK8EsdW8QRBqUDKk40uJjv/lFPS9Hebm4jz63u5vM6p05Xfauequfe1v5QVc+tq4jAzMxsb9uv6g6YmVnf5IAxM7MsHDBmZpaFA8bMzLJwwJiZWRb9q+5AT3HYYYfFiBEjqu6GmVmvsnz58l9HRFOtddkCRtJwYC4wGAhgVkTcJOlQ4B5gBLAeODci3pAk4CbgLOAd4IKIeCbtawrwV2nX34+IOal+HHAHcDCwALg8IqKzY3TV3xEjRtDS0rJXXruZ2b5C0iudrct5iWwHcGVEjAbGAtMljQZmAIsjYhSwOC0DjAdGpWkaMBMghcU1wAnA8cA1kgalNjOBi0rtxqV6Z8cwM7MGyRYwEbGp/QwkIt4GngOGAhOAOWmzOcDZaX4CMDcKS4GBkoYAZwKLImJrOgtZBIxL6wZExNIoPi06t8O+ah3DzMwapCE3+SWNAI4BngIGR8SmtOo1iktoUITPq6VmG1Ktq/qGGnW6OEbHfk2T1CKppa2trfsvzMzMOpU9YCQdAtwPXBER28rr0plH1mfVdHWMiJgVEc0R0dzUVPMelZmZ7aGsASNpf4pwuTMifpnKr6fLW6Sfm1N9IzC81HxYqnVVH1aj3tUxzMysQbIFTBoVdjvwXET8qLRqPjAlzU8BHizVJ6swFngrXeZaCJwhaVC6uX8GsDCt2yZpbDrW5A77qnUMMzNrkJyfgzkROB9YJWlFqv0lcANwr6SpwCvAuWndAoohyq0Uw5QvBIiIrZK+ByxL210XEVvT/CXsHKb8SJro4hhmZtYg8uP6C83NzeHPwZiZdY+k5RHRXGudHxVjZmZZ+FExZvaRPf65z1fdhb3u8088XnUXej2fwZiZWRYOGDMzy8IBY2ZmWThgzMwsCweMmZll4YAxM7MsHDBmZpaFA8bMzLJwwJiZWRYOGDMzy8IBY2ZmWThgzMwsCweMmZll4YAxM7MsHDBmZpaFA8bMzLLIFjCSZkvaLGl1qXaPpBVpWi9pRaqPkPRuad1fl9ocJ2mVpFZJN0tSqh8qaZGkF9PPQamutF2rpJWSjs31Gs3MrHM5z2DuAMaVCxHxnyJiTESMAe4Hflla/VL7uoi4uFSfCVwEjEpT+z5nAIsjYhSwOC0DjC9tOy21NzOzBssWMBHxBLC11rp0FnIucHdX+5A0BBgQEUsjIoC5wNlp9QRgTpqf06E+NwpLgYFpP2Zm1kBV3YM5CXg9Il4s1UZKelbS45JOSrWhwIbSNhtSDWBwRGxK868Bg0ttXu2kzS4kTZPUIqmlra3tI7wcMzPrqKqAmcSuZy+bgMMj4hjgL4C7JA2od2fp7Ca624mImBURzRHR3NTU1N3mZmbWhf6NPqCk/sBXgePaaxGxHdie5pdLegn4FLARGFZqPizVAF6XNCQiNqVLYJtTfSMwvJM2ZmbWIFWcwZwGPB8RH1z6ktQkqV+aP4LiBv26dAlsm6Sx6b7NZODB1Gw+MCXNT+lQn5xGk40F3ipdSjMzswbJOUz5buBXwJGSNkiamlZN5MM39z8HrEzDlu8DLo6I9gEClwA/A1qBl4BHUv0G4HRJL1KE1g2pvgBYl7a/LbU3M7MGy3aJLCImdVK/oEbtfophy7W2bwGOqlHfApxaox7A9G5218zM9jJ/kt/MzLJwwJiZWRYOGDMzy8IBY2ZmWThgzMwsCweMmZll4YAxM7MsHDBmZpaFA8bMzLJwwJiZWRYOGDMzy8IBY2ZmWThgzMwsCweMmZll4YAxM7MsHDBmZpaFA8bMzLJwwJiZWRbZAkbSbEmbJa0u1a6VtFHSijSdVVp3taRWSS9IOrNUH5dqrZJmlOojJT2V6vdIOiDVD0zLrWn9iFyv0czMOpfzDOYOYFyN+o8jYkyaFgBIGg1MBD6d2twqqZ+kfsAtwHhgNDApbQvwg7SvTwJvAFNTfSrwRqr/OG1nZmYNli1gIuIJYGudm08A5kXE9oh4GWgFjk9Ta0Ssi4j3gHnABEkCTgHuS+3nAGeX9jUnzd8HnJq2NzOzBqriHsylklamS2iDUm0o8Gppmw2p1ln948CbEbGjQ32XfaX1b6XtP0TSNEktklra2to++iszM7MPNDpgZgKfAMYAm4AbG3z8XUTErIhojojmpqamKrtiZtbnNDRgIuL1iHg/In4H3EZxCQxgIzC8tOmwVOusvgUYKKl/h/ou+0rr/yBtb2ZmDdTQgJE0pLT4FaB9hNl8YGIaATYSGAU8DSwDRqURYwdQDASYHxEBLAHOSe2nAA+W9jUlzZ8D/GPa3szMGqj/7jfZM5LuBk4GDpO0AbgGOFnSGCCA9cA3ACJijaR7gbXADmB6RLyf9nMpsBDoB8yOiDXpEFcB8yR9H3gWuD3Vbwd+IamVYpDBxFyv0czMOpctYCJiUo3y7TVq7dtfD1xfo74AWFCjvo6dl9jK9d8CX+tWZ83MbK/zJ/nNzCwLB4yZmWXhgDEzsywcMGZmloUDxszMsnDAmJlZFg4YMzPLwgFjZmZZOGDMzCwLB4yZmWXhgDEzsywcMGZmloUDxszMsnDAmJlZFg4YMzPLwgFjZmZZOGDMzCyLbAEjabakzZJWl2o/lPS8pJWSHpA0MNVHSHpX0oo0/XWpzXGSVklqlXSzJKX6oZIWSXox/RyU6krbtabjHJvrNZqZWedynsHcAYzrUFsEHBURRwP/DFxdWvdSRIxJ08Wl+kzgImBUmtr3OQNYHBGjgMVpGWB8adtpqb2ZmTVYtoCJiCeArR1qj0bEjrS4FBjW1T4kDQEGRMTSiAhgLnB2Wj0BmJPm53Soz43CUmBg2o+ZmTVQlfdgvg48UloeKelZSY9LOinVhgIbSttsSDWAwRGxKc2/BgwutXm1kza7kDRNUouklra2to/wUszMrKNKAkbSd4AdwJ2ptAk4PCKOAf4CuEvSgHr3l85uorv9iIhZEdEcEc1NTU3dbW5mZl3o3+gDSroA+CJwagoGImI7sD3NL5f0EvApYCO7XkYblmoAr0saEhGb0iWwzam+ERjeSRszM2uQhp7BSBoHfBv4ckS8U6o3SeqX5o+guEG/Ll0C2yZpbBo9Nhl4MDWbD0xJ81M61Cen0WRjgbdKl9LMzKxBsp3BSLobOBk4TNIG4BqKUWMHAovSaOOlacTY54DrJP0/4HfAxRHRPkDgEooRaQdT3LNpv29zA3CvpKnAK8C5qb4AOAtoBd4BLsz1Gs3MrHPZAiYiJtUo397JtvcD93eyrgU4qkZ9C3BqjXoA07vVWTMz2+v8SX4zM8vCAWNmZlk4YMzMLAsHjJmZZeGAMTOzLBwwZmaWhQPGzMyycMCYmVkWdQWMpMX11MzMzNp1+Ul+SQcBv0fxuJdBgNKqAXTyCHwzMzPY/aNivgFcAfwRsJydAbMN+GnGfpmZWS/XZcBExE3ATZK+GRE/aVCfzMysD6jrYZcR8RNJnwVGlNtExNxM/TIzs16uroCR9AvgE8AK4P1UDsABY2ZmNdX7uP5mYHT7N1CamZntTr2fg1kN/LucHTEzs76l3jOYw4C1kp4GtrcXI+LLWXplZma9Xr0Bc23OTpiZWd9T1yWyiHi81rS7dpJmS9osaXWpdqikRZJeTD8Hpbok3SypVdJKSceW2kxJ278oaUqpfpykVanNzZLU1THMzKxx6h1F9jbFqDGAA4D9gd9ExIDdNL2D4gOZ5dFmM4DFEXGDpBlp+SpgPDAqTScAM4ETJB0KXEMx0CCA5ZLmR8QbaZuLgKeABcA44JEujrFHjvsvfW+w3PIfTq66C2bWx9V7BvOxiBiQAuVg4M+BW+to9wSwtUN5AjAnzc8Bzi7V50ZhKTBQ0hDgTGBRRGxNobIIGJfWDYiIpWl029wO+6p1DDMza5BuP005BcDfUfzh3xODI2JTmn8NGJzmhwKvlrbbkGpd1TfUqHd1jF1ImiapRVJLW1vbHr4cMzOrpd5LZF8tLe5Hcbnqtx/14BERkrJ+tqarY0TELGAWQHNzsz/jY2a2F9U7iuxLpfkdwHqKy1B74nVJQyJiU7rMtTnVNwLDS9sNS7WNwMkd6o+l+rAa23d1DDMza5B678FcWJouiojrI2JP/2jPB9pHgk0BHizVJ6fRZGOBt9JlroXAGZIGpdFgZwAL07ptksam0WOTO+yr1jHMzKxB6v3CsWGSHkhDjjdLul/SsDra3Q38CjhS0gZJU4EbgNMlvQiclpahGAW2DmgFbgMuAYiIrcD3gGVpui7VSNv8LLV5iWIEGV0cw8zMGqTeS2Q/B+4CvpaWz0u107tqFBGTOll1ao1tA5jeyX5mA7Nr1FuAo2rUt9Q6htnedOJPTqy6C3vdk998suouWB9S7yiypoj4eUTsSNMdQFPGfpmZWS9Xb8BskXSepH5pOg/YkrNjZmbWu9UbMF8HzqX4TMkm4Bzggkx9MjOzPqDeezDXAVPSJ+lJj2/5HxTBY2Zm9iH1nsEc3R4u8MHIrmPydMnMzPqCegNmv/ITidMZTL1nP2Zmtg+qNyRuBH4l6W/T8teA6/N0yczM+oK6AiYi5kpqAU5Jpa9GxNp83TIzs96u7stcKVAcKmZmVpduP67fzMysHr5Rb93yL9d9puou7HWH/9dVVXfBrE/yGYyZmWXhgDEzsywcMGZmloUDxszMsnDAmJlZFg4YMzPLwgFjZmZZNDxgJB0paUVp2ibpCknXStpYqp9VanO1pFZJL0g6s1Qfl2qtkmaU6iMlPZXq90g6oNGv08xsX9fwgImIFyJiTESMAY4D3gEeSKt/3L4uIhYASBoNTAQ+DYwDbm3/Zk3gFmA8MBqYlLYF+EHa1yeBN4CpjXp9ZmZWqPoS2anASxHxShfbTADmRcT2iHgZaAWOT1NrRKyLiPeAecAESaJ4KOd9qf0c4Oxsr8DMzGqq+lExE4G7S8uXSpoMtABXpi85GwosLW2zIdUAXu1QPwH4OPBmROyosf0uJE0DpgEcfvjhH+2VmJkBP73yoaq7sNddeuOX9qhdZWcw6b7Il4H275iZCXwCGANsovgOmqwiYlZENEdEc1NTU+7DmZntU6o8gxkPPBMRrwO0/wSQdBvwcFrcCAwvtRuWanRS3wIMlNQ/ncWUtzczswap8h7MJEqXxyQNKa37CrA6zc8HJko6UNJIYBTwNLAMGJVGjB1AcbltfkQEsAQ4J7WfAjyY9ZWYmdmHVHIGI+n3gdOBb5TK/13SGCCA9e3rImKNpHspvuxsBzA9It5P+7kUWAj0A2ZHxJq0r6uAeZK+DzwL3J79RZmZ2S4qCZiI+A3Fzfhy7fwutr8euL5GfQGwoEZ9HcUoMzMzq0jVw5TNzKyPcsCYmVkWDhgzM8vCAWNmZlk4YMzMLAsHjJmZZeGAMTOzLBwwZmaWhQPGzMyycMCYmVkWDhgzM8vCAWNmZlk4YMzMLAsHjJmZZeGAMTOzLBwwZmaWhQPGzMyyqCxgJK2XtErSCkktqXaopEWSXkw/B6W6JN0sqVXSSknHlvYzJW3/oqQppfpxaf+tqa0a/yrNzPZdVZ/B/MeIGBMRzWl5BrA4IkYBi9MywHhgVJqmATOhCCTgGuAEiq9IvqY9lNI2F5Xajcv/cszMrF3VAdPRBGBOmp8DnF2qz43CUmCgpCHAmcCiiNgaEW8Ai4Bxad2AiFgaEQHMLe3LzMwaoMqACeBRScslTUu1wRGxKc2/BgxO80OBV0ttN6RaV/UNNeq7kDRNUouklra2to/6eszMrKR/hcf+s4jYKOkPgUWSni+vjIiQFDk7EBGzgFkAzc3NWY9lZravqewMJiI2pp+bgQco7qG8ni5vkX5uTptvBIaXmg9Lta7qw2rUzcysQSoJGEm/L+lj7fPAGcBqYD7QPhJsCvBgmp8PTE6jycYCb6VLaQuBMyQNSjf3zwAWpnXbJI1No8cml/ZlZmYNUNUlssHAA2nkcH/groj4B0nLgHslTQVeAc5N2y8AzgJagXeACwEiYquk7wHL0nbXRcTWNH8JcAdwMPBImszMrEEqCZiIWAf8SY36FuDUGvUApneyr9nA7Br1FuCoj9xZMzPbIz1tmLKZmfURDhgzM8vCAWNmZlk4YMzMLAsHjJmZZeGAMTOzLBwwZmaWhQPGzMyycMCYmVkWDhgzM8vCAWNmZlk4YMzMLAsHjJmZZeGAMTOzLBwwZmaWhQPGzMyycMCYmVkWDhgzM8ui4QEjabikJZLWSloj6fJUv1bSRkkr0nRWqc3VklolvSDpzFJ9XKq1SppRqo+U9FSq3yPpgMa+SjMzq+IMZgdwZUSMBsYC0yWNTut+HBFj0rQAIK2bCHwaGAfcKqmfpH7ALcB4YDQwqbSfH6R9fRJ4A5jaqBdnZmaFhgdMRGyKiGfS/NvAc8DQLppMAOZFxPaIeBloBY5PU2tErIuI94B5wARJAk4B7kvt5wBn53k1ZmbWmUrvwUgaARwDPJVKl0paKWm2pEGpNhR4tdRsQ6p1Vv848GZE7OhQr3X8aZJaJLW0tbXthVdkZmbtKgsYSYcA9wNXRMQ2YCbwCWAMsAm4MXcfImJWRDRHRHNTU1Puw5mZ7VP6V3FQSftThMudEfFLgIh4vbT+NuDhtLgRGF5qPizV6KS+BRgoqX86iylvb2ZmDVLFKDIBtwPPRcSPSvUhpc2+AqxO8/OBiZIOlDQSGAU8DSwDRqURYwdQDASYHxEBLAHOSe2nAA/mfE1mZvZhVZzBnAicD6yStCLV/pJiFNgYIID1wDcAImKNpHuBtRQj0KZHxPsAki4FFgL9gNkRsSbt7ypgnqTvA89SBJqZmTVQwwMmIv4PoBqrFnTR5nrg+hr1BbXaRcQ6ilFmZmZWEX+S38zMsnDAmJlZFg4YMzPLwgFjZmZZOGDMzCwLB4yZmWXhgDEzsywcMGZmloUDxszMsnDAmJlZFg4YMzPLwgFjZmZZOGDMzCwLB4yZmWXhgDEzsywcMGZmloUDxszMsnDAmJlZFn02YCSNk/SCpFZJM6ruj5nZvqZPBoykfsAtwHhgNDBJ0uhqe2Vmtm/pkwEDHA+0RsS6iHgPmAdMqLhPZmb7FEVE1X3Y6ySdA4yLiP+cls8HToiISztsNw2YlhaPBF5oaEdrOwz4ddWd6CH8XhT8Puzk92KnnvJe/HFENNVa0b/RPelJImIWMKvqfpRJaomI5qr70RP4vSj4fdjJ78VOveG96KuXyDYCw0vLw1LNzMwapK8GzDJglKSRkg4AJgLzK+6Tmdk+pU9eIouIHZIuBRYC/YDZEbGm4m7Vq0ddsquY34uC34ed/F7s1OPfiz55k9/MzKrXVy+RmZlZxRwwZmaWhQOmB5A0XNISSWslrZF0edV9ajRJ/SQ9K+nhqvvSU0g6SNLTkv4p/Xfx3ar7VBVJsyVtlrS66r70BL3lUVgOmJ5hB3BlRIwGxgLT98FH21wOPFdrhaT1je1Kj7EdOCUi/gQYA4yTNLbiPlXlDmBc1Z3oCXrTo7AcMD1ARGyKiGfS/NsUf2iHVturxpE0DPgC8LOq+9KTROHf0uL+adonR+VExBPA1qr70UP0mkdhOWB6GEkjgGOAp6rtSUP9T+DbwO+q7khPky4drgA2A4siYl/678JqGwq8WlreQA/9B6kDpgeRdAhwP3BFRGyruj+NIOmLwOaIWN6hfoukFemP6x+1z0v6TjU9rUZEvB8RYyieRnG8pKOq7pNZvfrkBy17I0n7U4TLnRHxy6r700AnAl+WdBZwEDBA0t9ExHntG0han/7I7rMi4k1JSyjuQ/hG976t1zwKy2cwPYAkAbcDz0XEj6ruTyNFxNURMSwiRlA80ucfy+GyL5PUJGlgmj8YOB14vtpeWQ/Qax6F5YDpGU4EzgdOKV0KOqvqTlnlhgBLJK2k+KOyKCL2yWHcku4GfgUcKWmDpKlV96kqEbEDaH8U1nPAvT31UVh+VIyZmWXhMxgzM8vCAWNmZlk4YMzMLAsHjJmZZeGAMTOzLBwwZmaWhQPGejRJ30mPql+ZPh90gqQrJP3eXj7OekmH7WHbb0l6PvVvmaTJe7NvdRx/oKRL6tjuU5IWSHpR0jOS7pU0WFKzpJvTNidL+uwe9OEiSfeUlgdIeknSEd3dl/UdDhjrsST9KfBF4NiIOBo4jeIhf1cAezVg9pSkiyk+YX98epzNqYC60b5/V8t1Ggh0GTCSDgL+HpgZEaMi4ljgVqApIloi4rK06clAtwOG4knYwyWdlpavA2ZHxLo92Fd7n/vtaVvrISLCk6ceOQFfBR7qULsMeA9YBSxJtZlAC7AG+G6qnQL8Xand6cADXRxrPXAYMILi09G3pf09ChzcRbt/AY7oap9pvhl4LM1fC/wCeBK4u8ZyE8Vz6Zal6cRSu9nAY8A64LJUnwe8C6wAfthJX74OzO1k3cnAw+m1v0bxXKsVwEnAy8D+absB5eUa+zma4jlpzen3cwBwHvB02t//Avp19jsrvWc/AJ6heATKZcBaYCUwr+r/Jj11b6q8A548dTYBh6Q/TP9M8a/tz6f6B3+40/Kh6We/9Mf3aIqziOcp/oUOcBfwpS6OVQ6YHcCYVL8XOK+TNgOAN3a3zzTfMWCWk4KrxvJdwJ+l+cMpnlHXvt3/BQ5Mfd1C8R0xI4DVu3kvfwRc3sm6k4GHS8f4Vmndz4Gz0/w04MbdHOdG4K20z38PPFQKqFuByZ39zkrv2bdL+/tX4MA0P7Dq/yY9dW/yJTLrsaL4sq3jKP6wtQH3SLqgxqbnSnoGeBb4NDA6ir9IvwDOSw+M/FPgkToP/XJErEjzyyn+gO9t8yPi3U6WTwN+mr6qYD7FE6YPSev+PiK2R8SvKb4jZnCGvpX9DLgwzV9IEThduQXYGBGPUVwuPA5Yll7LqUD7PZkP/c5K+7inNL8SuFPSeRTBb72IH9dvPVpEvE/xL9zHJK0CppTXSxoJfAv4DxHxhqQ7KB77D8Ufw4eA3wJ/G8VDAuuxvTT/PnBwJ33bJunfJB0Rte817GDnfc6DOqz7TRfL+wFjI+K35Q2Kh25/qG/1/j+8Bvh8ndt+ICKelDRC0skUl7d291UBv2PnF8cJmBMRV5c32M3vDHZ9L74AfA74EvAdSZ/pxu/RKuYzGOuxJB0paVSpNAZ4BXgb+FiqDaD4g/SWpMEU31MOQET8K8Ullr9i9//y3lP/DbhF0oDU50NKo8jWU/wLHuDPu7HPR4Fvti9I2t134ZTfj87cBXxW0hdK+/1cjS8wq7Wvual9d9/DxcA5kv4wHe9QSX9MF7+zMkn7AcMjYglwFfAHFJdNrZdwwFhPdggwR9La9Mj60RT3CGYB/yBpSUT8E8Vllucp/gg+2WEfdwKvRsRzmfo4E1hCcRloNfC/2fkv+O8CN0lqoTjbqNdlQHMamr0WuLirjSNiC/CkpNWSftjJNu9SjMj7ZhqmvJZi5Flbh00fAr6ShlyflGp3AoMoBiDULSLWUoT7o+n3twgYUsfvrF0/4G/SmeuzwM0R8WZ3+mDV8uP6rU+T9FPg2Yi4veq+9FaSzgEmRMT5VffFehffg7E+S9JyiksxV1bdl95K0k8oLmH5C/Cs23wGY/sUSU9RDPMtOz8iVu2m3S0U3zxadlNE5Lq3s0ckfYZi9FzZ9og4YS8eo1e8F1Y9B4yZmWXhm/xmZpaFA8bMzLJwwJiZWRYOGDMzy+L/AwjE+zohzasXAAAAAElFTkSuQmCC\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.groupby(\"Stay_In_Current_City_Years\").mean()[\"Purchase\"].plot(kind='bar')\n",
        "plt.title(\"Stay_In_Current_City_Years and Purchase Analysis\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 301
        },
        "id": "yCF0Ty92_FOx",
        "outputId": "4ca30419-ef73-4333-c2f3-a14198c6e64c"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAX0AAAEcCAYAAAAr0WSuAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAdUElEQVR4nO3debgcVZ3/8feHhB1DgMQISSAwRB0QRIgBxYUBlE0NKjiMAmEzw8imoyL89BlQQXAcB0EBByUSEAXELQoqPEh0EA0EyABJQGJYQiAYyMYOge/vj3NubnHpvrf7ptN9w/m8nqef23XqVNWp5X66+lR1tyICMzMrw1qdboCZmbWPQ9/MrCAOfTOzgjj0zcwK4tA3MyuIQ9/MrCAOfRuQJL1b0r2dbseaRNI0SceU3obVSdIlks5YxXn8RtLEVrWpWUWEvqR3SbpZ0jJJiyX9SdLb87gjJN3UhjaEpG1bMJ+PS5oh6SlJj+YD6F2taOOqanZbShov6VpJS/N+uUXSkQAR8b8R8aZK3Qck7b0KbZOkP0o6rUf54ZL+JmmD/s57TSDpdEkv5uNmaf5/eEen27W65P09T9LsTrelp4jYLyKmdGr5r/nQlzQE+DXwbWBTYCTwZeD5TrarPyT9O/At4GvACGBL4AJgQj/mNbiRstUlB87vgT8A2wKbAf8G7Lc6lhfpU4jHAJ+RtH1uw3Dgm8AxEfFMK5bTzm3YD1dGxEbAcOAm4GeS1MwMBvj6Vb0HeD2wTdcJnmUR8Zp+AOOApXXG/SPwHPAS8FRXPeAA4A5gOTAfOL0yzTXACT3mcyfw4T7aEcC2+fnpwFXApcCTwCxgXB/Tb5zbeHAvdS4BzqgM7wE8XBl+APhCbu/zpLAN4GjgIeCPud5RwBxgCfA7YKse63EscB+wFDgfUL1t2UtbbwLO72X8yrYDlwEvA8/meZ+8Cvvh88BfSCc8Pwa+m8s/AMzM63QzsGNlmlOAv+V9Nbu6DOAI4E/AOcATwBl5u/4BWAY8Tgrbeu35CbAw1/0jsH2P/Xl+XtcngenAP1TGvw+4J0/7nbzMY+os53Tgh5Xh7fO+HAZMq06X1+mmHvv8uLzP789lE/L2Wp63zb65fBrw1bxNngSuA4Y1uL775+37JLAA+FxlXN39U2d9JwOXAz8DvtNj3Kq08RLy/xhwN/DByri18/5+G7Ae8MN8TCwFbgVGVJZ/TH7e8LHSqkfbQ7jdD2BI3vBTSGeRm/QY/4oDPJftAexACoYdgceAA/O4jwHTK3Xfmue/Th/t6Bn6z+WDfBBwFvCXPqbfF1gBDO6lzsoDsrIePUN/JjAaWB8Yk9t1KbBhLpsAzCWF+GDgS8DNPdbj18BQ0juNRXT/w79qW9Zp5wakF4d/6qVOrbbvXRnu734YRArPn5Fe6F6X/0n/Duyax0/My1s3T3MwsEU+Hv4ZeBrYvLLOK4AT8vZan/Ri8sVcfz3gXb2056jchnVJ7+Jm9tifTwDj87wvB67I44aRAusgUth8Jrejz9DPy/oG8FAenkbfoX896Z3y+rk9y0gvOmuR3j2/uTKvvwFvzHWnAWc3uL6PAu/OzzcBds7Pe90/dY6v5aT/r4+SwnSdyvhVaeMldIf+yVRCmvS/c1d+/q/Ar3JbBgG7AEN6bu9mjpVWPV7z3TsRsRx4F+nA/R6wSNJUSSN6mWZaRNwVES9HxJ2kHfPePHoq8EZJY/PwYaQd/0KTTbspIq6NiJdIZ7Jv7aP+ZsDjEbGiyeX0dF5EzI+IZytlp0fE07nsWOCsiJiTl/U1YCdJW1Xqnx0RSyPiIeBGYKcm27AJ6SB/dBXWo1/7IW/vo4APk94pPAlMAv4nIqZHxEuR+lufB3bL0/wkIh7Jx8OVpDPe8ZXZPhIR346IFXkbvghsBWwREc9FRN3rHBExOSKejIjnScH8VkkbV6r8PCJuyfvicrq39f7ArIi4OiJeJIXTwt43GR+TtJT07nWXvA0adVZELM7rdzQwOSKuz9tkQUTcU6n7g4j4a657VaXNfa3vi8B2koZExJKIuD2X97p/avhIHn8d6V3S2qR371X9bWPVD4H9cxcypGPwssq6bEY60XspIm7LWdRTw8dKq7zmQx8gB9gRETEKeAvprO1b9epL2lXSjZIWSVpGCsJheV7PAVcCh0paC/gXund0M6r/oM8A6/XRX/oEMKwFfarz+yjbCjg3X+xbCiwmdd+MrNTp2faNmmzDElJ3zeZNTrfSquyHiJiVn3b93Qr4bNc65/UeTTpOui72zqyMewv5eMh6btOTSdvsFkmzJB1Vqx2SBkk6O19IXk46e6XHvOtt6y2qy4102lhr31ZdFRFDI+L1EbFnRNzWR/2q6rxHk86U66nZ5gbW96OkF7MHJf2hcqG51/1Tw0TSuq7Ix8lPc1kr2rhSRDxC6iL6qKShpJ6Ey/Poy0hdo1dIekTSf0pau0ZbGzpWWqmI0K/KZySXkP5xIb0D6OlHpDPJ0RGxMfBd0o7pMgX4BLAX8ExE/Hm1Nbjbn0lnLwf2Uudp0tvJLm+oUafW+lbL5gP/msOh67F+RNzcQBsb+srWSBdN/0z6J29UrXm3aj/MB87ssc4bRMSP8zuc7wHHA5tFxFBSX271eHhF2yJiYUR8MiK2IL3Nv6DOnVsfJ3UJ7E26ZjMmlzdycfVRUvClCdIF2dH1q/eq2eNmPvAP/VhOr+sbEbdGxATSBdhfkM7Au5ZXc//0XICkUcCepJOBhZIWkrrA9pf0quButo01TAEOJXUB/jkiFuR1eTEivhwR2wHvJF2TOLznxE0cKy3zmg99SW+W9Nl8MCBpNOms8C+5ymPAKEnrVCZ7HbA4Ip6TNJ50IKyUw+Vl0p0f/TnLb1pELAP+Azhf0oGSNpC0tqT9JP1nrjaTdHBvKukNwKf7sajvAqdW7nDZWNLBDU5ba1vWczJwhKTPS9osL+utkq7oZd7bVAtauB++Bxyb3+FJ0oaSDpD0OtK1jiBdu0DpltK39DIvJB3cdbyR3tVEbmdPryO9kD9BCt2vNdHma4DtJX0kv/s7kdph3YiZwEfyMbUtqfumNxcDR0raS9JakkZKenMDy6m7vpLWkfQJSRvn7qrldG+z3vZPT4cBfwXeROqy2YnUd/8w6f++322s4xfAzsBJpGtjXevzT5J2kDQor8uL1DgGmjhWWuY1H/qki127AtMlPU0K+7uBz+bxvye9zV8o6fFc9ingK5KeJAXtVbzapaSLvT9cjW1/hYj4JvDvpIuri0hnQMeTDjxIwfd/pLek15G6P5pdxs+Br5Peli4nbatGb6OstS3rLedm0hnZnsA8SYuBi4Br60xyFvCl/Pb+c5XyVd4PETED+CTpDpglpAvZR+Rxs0kvKn8mvfDsQHpL35u3k463p0jvGE+KiHk16l0KPEi6U2U23ScijbT5cdLZ5dmkgBrbQLvqOQd4gbR+U+juoqi37FuAI/N0y0h3n2zV2zRZX+t7GPBAPu6OJb2L63X/1DARuCCfQa98kE5mGvlAVFP7JF8T+CmwNenmgC5vAK4mBf4c0jaqdWLS6LHSMkpdgdYsSYcDkyJiQHwwqlTeD9Zpkv4DeGNEHNrptjRiTfmgxYCi9OnNT5E+GGUd4v1gnSZpU1J32GGdbkujSujeaSlJ+5C6Vh4jXfDtKn+30kfcX/VoYt5b1puHpC1Xw+qsVvluhFrr8okWzHu17QezRkj6JKmL9TcR8cdOt6dR7t4xMyuIz/TNzAri0DczK8iAvpA7bNiwGDNmTKebYWa2Rrntttsej4jhtcYN6NAfM2YMM2bM6HQzzMzWKJIerDfO3TtmZgVx6JuZFcShb2ZWEIe+mVlBHPpmZgVx6JuZFcShb2ZWEIe+mVlBBvSHs8xWlzGnXNPpJvDA2T1/q9ts9XPomxXOL4DdStgWDv2CDIQDGgbOP7hZiV7zoe+gMzPr5gu5ZmYFceibmRXEoW9mVhCHvplZQRz6ZmYFceibmRXEoW9mVhCHvplZQRz6ZmYFceibmRXEoW9mVhCHvplZQRz6ZmYFceibmRXEoW9mVhCHvplZQRz6ZmYFceibmRXEoW9mVhCHvplZQRz6ZmYFceibmRWkodCX9BlJsyTdLenHktaTtLWk6ZLmSrpS0jq57rp5eG4eP6Yyn1Nz+b2S9lk9q2RmZvX0GfqSRgInAuMi4i3AIOAQ4OvAORGxLbAEODpPcjSwJJefk+shabs83fbAvsAFkga1dnXMzKw3jXbvDAbWlzQY2AB4FNgTuDqPnwIcmJ9PyMPk8XtJUi6/IiKej4j7gbnA+FVfBTMza1SfoR8RC4D/Ah4ihf0y4DZgaUSsyNUeBkbm5yOB+XnaFbn+ZtXyGtOYmVkbNNK9swnpLH1rYAtgQ1L3zGohaZKkGZJmLFq0aHUtxsysSI107+wN3B8RiyLiReBnwO7A0NzdAzAKWJCfLwBGA+TxGwNPVMtrTLNSRFwUEeMiYtzw4cP7sUpmZlZPI6H/ELCbpA1y3/xewGzgRuCgXGci8Mv8fGoeJo//fURELj8k392zNTAWuKU1q2FmZo0Y3FeFiJgu6WrgdmAFcAdwEXANcIWkM3LZxXmSi4HLJM0FFpPu2CEiZkm6ivSCsQI4LiJeavH6mJlZL/oMfYCIOA04rUfxPGrcfRMRzwEH15nPmcCZTbbRzMxaxJ/INTMriEPfzKwgDn0zs4I49M3MCuLQNzMriEPfzKwgDn0zs4I49M3MCuLQNzMriEPfzKwgDn0zs4I49M3MCuLQNzMriEPfzKwgDn0zs4I49M3MCuLQNzMriEPfzKwgDn0zs4I49M3MCuLQNzMriEPfzKwgDn0zs4I49M3MCuLQNzMriEPfzKwgDn0zs4I49M3MCuLQNzMriEPfzKwgDn0zs4I49M3MCuLQNzMriEPfzKwgDn0zs4I0FPqShkq6WtI9kuZIeoekTSVdL+m+/HeTXFeSzpM0V9KdknauzGdirn+fpImra6XMzKy2Rs/0zwV+GxFvBt4KzAFOAW6IiLHADXkYYD9gbH5MAi4EkLQpcBqwKzAeOK3rhcLMzNqjz9CXtDHwHuBigIh4ISKWAhOAKbnaFODA/HwCcGkkfwGGStoc2Ae4PiIWR8QS4Hpg35aujZmZ9aqRM/2tgUXADyTdIen7kjYERkTEo7nOQmBEfj4SmF+Z/uFcVq/czMzapJHQHwzsDFwYEW8Dnqa7KweAiAggWtEgSZMkzZA0Y9GiRa2YpZmZZY2E/sPAwxExPQ9fTXoReCx325D//j2PXwCMrkw/KpfVK3+FiLgoIsZFxLjhw4c3sy5mZtaHPkM/IhYC8yW9KRftBcwGpgJdd+BMBH6Zn08FDs938ewGLMvdQL8D3i9pk3wB9/25zMzM2mRwg/VOAC6XtA4wDziS9IJxlaSjgQeBj+W61wL7A3OBZ3JdImKxpK8Ct+Z6X4mIxS1ZCzMza0hDoR8RM4FxNUbtVaNuAMfVmc9kYHIzDTQzs9bxJ3LNzAri0DczK4hD38ysIA59M7OCOPTNzAri0DczK4hD38ysIA59M7OCOPTNzAri0DczK4hD38ysIA59M7OCOPTNzAri0DczK4hD38ysIA59M7OCOPTNzAri0DczK4hD38ysIA59M7OCOPTNzAri0DczK4hD38ysIA59M7OCOPTNzAri0DczK4hD38ysIA59M7OCOPTNzAri0DczK4hD38ysIA59M7OCOPTNzAri0DczK4hD38ysIA59M7OCNBz6kgZJukPSr/Pw1pKmS5or6UpJ6+TydfPw3Dx+TGUep+byeyXt0+qVMTOz3jVzpn8SMKcy/HXgnIjYFlgCHJ3LjwaW5PJzcj0kbQccAmwP7AtcIGnQqjXfzMya0VDoSxoFHAB8Pw8L2BO4OleZAhyYn0/Iw+Txe+X6E4ArIuL5iLgfmAuMb8VKmJlZYxo90/8WcDLwch7eDFgaESvy8MPAyPx8JDAfII9fluuvLK8xzUqSJkmaIWnGokWLmlgVMzPrS5+hL+kDwN8j4rY2tIeIuCgixkXEuOHDh7djkWZmxRjcQJ3dgQ9J2h9YDxgCnAsMlTQ4n82PAhbk+guA0cDDkgYDGwNPVMq7VKcxM7M26PNMPyJOjYhRETGGdCH29xHxCeBG4KBcbSLwy/x8ah4mj/99REQuPyTf3bM1MBa4pWVrYmZmfWrkTL+eLwBXSDoDuAO4OJdfDFwmaS6wmPRCQUTMknQVMBtYARwXES+twvLNzKxJTYV+REwDpuXn86hx901EPAccXGf6M4Ezm22kmZm1hj+Ra2ZWEIe+mVlBHPpmZgVx6JuZFcShb2ZWEIe+mVlBHPpmZgVx6JuZFcShb2ZWEIe+mVlBHPpmZgVx6JuZFcShb2ZWEIe+mVlBHPpmZgVx6JuZFcShb2ZWEIe+mVlBHPpmZgVx6JuZFcShb2ZWEIe+mVlBHPpmZgVx6JuZFcShb2ZWEIe+mVlBHPpmZgVx6JuZFcShb2ZWEIe+mVlBHPpmZgVx6JuZFcShb2ZWEIe+mVlBHPpmZgXpM/QljZZ0o6TZkmZJOimXbyrpekn35b+b5HJJOk/SXEl3Stq5Mq+Juf59kiauvtUyM7NaGjnTXwF8NiK2A3YDjpO0HXAKcENEjAVuyMMA+wFj82MScCGkFwngNGBXYDxwWtcLhZmZtUefoR8Rj0bE7fn5k8AcYCQwAZiSq00BDszPJwCXRvIXYKikzYF9gOsjYnFELAGuB/Zt6dqYmVmvmurTlzQGeBswHRgREY/mUQuBEfn5SGB+ZbKHc1m9cjMza5OGQ1/SRsBPgU9HxPLquIgIIFrRIEmTJM2QNGPRokWtmKWZmWUNhb6ktUmBf3lE/CwXP5a7bch//57LFwCjK5OPymX1yl8hIi6KiHERMW748OHNrIuZmfWhkbt3BFwMzImI/66Mmgp03YEzEfhlpfzwfBfPbsCy3A30O+D9kjbJF3Dfn8vMzKxNBjdQZ3fgMOAuSTNz2f8DzgauknQ08CDwsTzuWmB/YC7wDHAkQEQslvRV4NZc7ysRsbgla2FmZg3pM/Qj4iZAdUbvVaN+AMfVmddkYHIzDTQzs9bxJ3LNzAri0DczK4hD38ysIA59M7OCOPTNzAri0DczK4hD38ysIA59M7OCOPTNzAri0DczK4hD38ysIA59M7OCOPTNzAri0DczK4hD38ysIA59M7OCOPTNzAri0DczK4hD38ysIA59M7OCOPTNzAri0DczK4hD38ysIA59M7OCOPTNzAri0DczK4hD38ysIA59M7OCOPTNzAri0DczK4hD38ysIA59M7OCOPTNzAri0DczK4hD38ysIA59M7OCtD30Je0r6V5JcyWd0u7lm5mVrK2hL2kQcD6wH7Ad8C+StmtnG8zMStbuM/3xwNyImBcRLwBXABPa3AYzs2IpItq3MOkgYN+IOCYPHwbsGhHHV+pMAiblwTcB97atgfUNAx7vdCMGCG+Lbt4W3bwtug2EbbFVRAyvNWJwu1vSl4i4CLio0+2okjQjIsZ1uh0DgbdFN2+Lbt4W3Qb6tmh3984CYHRleFQuMzOzNmh36N8KjJW0taR1gEOAqW1ug5lZsdravRMRKyQdD/wOGARMjohZ7WxDPw2o7qYO87bo5m3Rzdui24DeFm29kGtmZp3lT+SamRXEoW9mVhCHvplZQQbcffqdJunNpE8Jj8xFC4CpETGnc62yTsvHxUhgekQ8VSnfNyJ+27mWtZ+k8UBExK35a1T2Be6JiGs73DRrgM/0KyR9gfTVEAJuyQ8BP/aXw72SpCM73YZ2kXQi8EvgBOBuSdWvDvlaZ1rVGZJOA84DLpR0FvAdYEPgFElf7GjjOkzSqZ1uQyN8906FpL8C20fEiz3K1wFmRcTYzrRs4JH0UERs2el2tIOku4B3RMRTksYAVwOXRcS5ku6IiLd1tIFtlLfFTsC6wEJgVEQsl7Q+6V3Qjh1tYAdJuj0idu50O/ri7p1XehnYAniwR/nmeVxRJN1ZbxQwop1t6bC1urp0IuIBSXsAV0vairQtSrIiIl4CnpH0t4hYDhARz0oq7n9kTeTQf6VPAzdIug+Yn8u2BLYFjq871WvXCGAfYEmPcgE3t785HfOYpJ0iYiZAPuP/ADAZ2KGzTWu7FyRtEBHPALt0FUramDJPjO4HgvQ/sbmkefl5RMQ2HW1cHe7e6UHSWqSvgK5eyL01n90URdLFwA8i4qYa434UER/vQLPaTtIo0hnuwhrjdo+IP3WgWR0had2IeL5G+TBg84i4qwPNGhDWlK4+h76ZWQusKaHvu3fMzFpjjXjH5zN9M7MWkXRpRBze6Xb0xhdyzcz6QVKtr4XfU9JQgIj4UJub1BCHvplZ/4wCZgPfp/sOnrcD3+xko/ri7h0zs37Id/qdBOwPfD4iZkqaN1Bv1ezi0DczWwX5lt5zgMeADw30T6q7e8fMbBVExMPAwZIOAJZ3uj198Zm+mVlBfJ++mVlBHPpmZgVx6JuZFcShb02T9EVJsyTdKWmmpF0lfVrSBi1ezgP5i7z6M+3nJN2T23erpLZ+SlLSUEmfaqDeGyVdK+k+SbdLukrSCEnjJJ2X6+wh6Z39aMMnJV1ZGR4i6W+SBvQthbZ6OfStKZLeAXwA2Dn/YMbepK+h/jTQ0tDvL0nHAu8DxkfETsBeNPG995IG9zbcoKFAr6EvaT3gGuDCiBibf4DjAmB4RMyIiBNz1T2ApkOf9KGh0ZL2zsNfASZHxLx+zKurzYP6O60NEBHhhx8NP4CPAL/qUXYi8AJwF3BjLrsQmAHMAr6cy/YEflGZ7n3Az3tZ1gPAMGAMMAf4Xp7fdcD6vUz3ELBNb/PMz8cB0/Lz04HLSF+a9eMaw8OBnwK35sfulekmA9OAecCJufwK4FlgJvCNOm05Cri0zrg9gF/ndV9I+orvmcC7gfuBtXO9IdXhGvPZEbg7r+tdwDrAoaSfAp0J/A8wqN4+q2yzrwO3A4fk/T0buBO4otPHpB/NPTreAD/WrAewUQ6Lv5LOSt+by1eGaR7eNP8dlANxR9LZ9j2kM1mAHwEf7GVZ1dBfAeyUy68CDq0zzRBgSV/zzM97hv5t5BeTGsM/At6Vn28JzKnUu5n084HDgCeAtXOb7+5jW/43cFKdcXsAv64s43OVcT8ADszPJwHf7GM53wSW5Xn+I/CryovGBcDh9fZZZZudXJnfI8C6+fnQTh+TfjT3cPeONSXSzwbuQgqbRcCVko6oUfVjkm4H7gC2B7aLlBKXAYfmL6V6B/CbBhd9f+RfriKF8Zh+r0R9UyPi2TrDewPfkTQTmAoMkbRRHndNRDwfEY8Df2f1/5Tk94GuH6Y/kvQi0JvzgQURMY3U1bULcGtel72Arj7+V+2zyjyurDy/E7hc0qGkF2Nbg/gTuda0SL8iNg2Yln8oe2J1vKStgc8Bb4+IJZIuAdbLo39AOtN8DvhJRDQaGtVfa3oJWL9O25ZLekrSNlG773oF3dey1usx7ulehtcCdouI56oVJNVqW6P/V7OA9zZYd6WI+JOkMfm3egdFxN19TPIy3T9lKGBKRJxardDHPoNXbosDgPcAHwS+KGmHJvajdZjP9K0pkt4kaWylaCfSD8k/Cbwulw0hhcQySSOA/boqR8QjpO6BL9H3GWp/nQWcL2lIbvNGlbt3HqD7t10/2sQ8rwNO6BqQtFMf9avbo54fAe/MH9/vmu97JL2lgXldmqdvdhveABwk6fV5eZvmH3ivu8+q8peMjY6IG4EvABuTuvxsDeHQt2ZtBEyRNFvSnaQugNOBi4DfSroxIv6P1EVwDymYev6i0OXA/IiYs5raeCFwI6kL427gf+k+0/0ycK6kGaSz8kadCIzLt6nOBo7trXJEPAH8SdLdkr5Rp86zpDuhTsi3bM4m3fGzqEfVXwEfzrefvjuXXQ5sQrrI3LCImE16wb0u77/rSb9t29c+6zII+GF+h3cHcF5ELG2mDdZZ/u4daztJ3wHuiIiLO92WNZWkg4AJEXFYp9tiaxb36VtbSbqN1I3w2U63ZU0l6duk7pf9O90WW/P4TN86TtJ00i2PVYdFxF19THc+sHuP4nMjYnVdK+gXSTuQ7lqqej4idm3hMtaIbWGd59A3MyuIL+SamRXEoW9mVhCHvplZQRz6ZmYFceibmRXk/wN/lAKgWVPXuAAAAABJRU5ErkJggg==\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.groupby(\"Age\").mean()[\"Purchase\"].plot(kind='bar')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 319
        },
        "id": "HmkIVyqg_50y",
        "outputId": "a8984729-4c2f-4544-f808-f28d454a70ef"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7fc96c812130>"
            ]
          },
          "metadata": {},
          "execution_count": 28
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYMAAAEdCAYAAADuCAshAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAXkUlEQVR4nO3dfbRddX3n8fdHAshDJQHTFBMqVFGKfVBIAUdHHVAIaAUrunC1krFoOjP4UDvLEWe6mjUiXdjRMjoKbSooYEekFIUqymQh0Ok4PCSAIFIlAyLJ4iE1EbRYNfqdP/bvwuF6Q+5Dcvc5ue/XWnfdfX5773O+N7n3fM7+7d/+7VQVkqS57Wl9FyBJ6p9hIEkyDCRJhoEkCcNAkoRhIEliEmGQ5IIkDyf5+kDbvklWJ7m7fV/Q2pPko0nWJbk9yWED+yxv29+dZPlA++FJ7mj7fDRJtvcPKUl6apM5MvgUsGxc2xnANVV1MHBNewxwPHBw+1oBnAddeAArgSOBI4CVYwHStnnbwH7jX0uStINtMwyq6u+BTeOaTwQubMsXAicNtF9UnRuA+Un2B44DVlfVpqraDKwGlrV1z6iqG6q7+u2igeeSJM2SedPcb1FVPdCWHwQWteXFwP0D261vbU/Vvn6C9gklWUF3xMFee+11+CGHHDLN8iVp7lm7du0/VdXCidZNNwweV1WVZFbmtKiqVcAqgKVLl9aaNWtm42UlaaeQ5L6trZvuaKKHWhcP7fvDrX0DcMDAdkta21O1L5mgXZI0i6YbBlcCYyOClgNXDLSf2kYVHQU80rqTrgaOTbKgnTg+Fri6rXs0yVFtFNGpA88lSZol2+wmSvIZ4BXAM5OspxsVdDZwaZLTgPuAN7bNrwJOANYBjwFvAaiqTUnOBG5u272/qsZOSv8HuhFLewBfal+SpFmUUZ3C2nMGkjQ1SdZW1dKJ1nkFsiTJMJAkGQaSJAwDSRKGgSSJ7XAFsiTtSAee8cUd+vzfPvvVO/T5R4VHBpIkw0CSZBhIkvCcgbRN9llrJnbk78/2/N3xyECS5JGBdjw/WUvDzzCQdnKGsSbDMBgB/jFL2tE8ZyBJMgwkSYaBJAnDQJLEHDqBPCoXfkhSHzwykCQZBpIkw0CShGEgScIwkCRhGEiSMAwkSRgGkiQMA0kShoEkCcNAkoRhIEnCMJAkYRhIkjAMJEkYBpIkZhgGSd6d5M4kX0/ymSRPT3JQkhuTrEvy2SS7tW13b4/XtfUHDjzP+1r7N5McN7MfSZI0VdMOgySLgXcCS6vq14BdgFOADwLnVNVzgc3AaW2X04DNrf2cth1JDm37vQBYBpybZJfp1iVJmrqZdhPNA/ZIMg/YE3gAOBq4rK2/EDipLZ/YHtPWH5Mkrf2SqvpRVd0LrAOOmGFdkqQpmHYYVNUG4EPAd+hC4BFgLfC9qtrSNlsPLG7Li4H7275b2vb7DbZPsM+TJFmRZE2SNRs3bpxu6ZKkcWbSTbSA7lP9QcCzgL3ounl2mKpaVVVLq2rpwoULd+RLSdKcMpNuolcC91bVxqr6CXA58BJgfus2AlgCbGjLG4ADANr6fYDvDrZPsI8kaRbMJAy+AxyVZM/W938M8A3gWuDkts1y4Iq2fGV7TFv/laqq1n5KG210EHAwcNMM6pIkTdG8bW8ysaq6McllwC3AFuBWYBXwReCSJB9obee3Xc4HLk6yDthEN4KIqrozyaV0QbIFOL2qfjrduiRJUzftMACoqpXAynHN9zDBaKCq+hfgDVt5nrOAs2ZSiyRp+rwCWZJkGEiSDANJEoaBJAnDQJKEYSBJwjCQJGEYSJIwDCRJGAaSJAwDSRKGgSQJw0CShGEgScIwkCRhGEiSMAwkSRgGkiQMA0kShoEkCcNAkoRhIEnCMJAkYRhIkjAMJEkYBpIkDANJEoaBJAnDQJKEYSBJwjCQJGEYSJIwDCRJGAaSJGYYBknmJ7ksyT8muSvJi5Psm2R1krvb9wVt2yT5aJJ1SW5PctjA8yxv29+dZPlMfyhJ0tTM9MjgI8CXq+oQ4DeBu4AzgGuq6mDgmvYY4Hjg4Pa1AjgPIMm+wErgSOAIYOVYgEiSZse0wyDJPsDLgPMBqurHVfU94ETgwrbZhcBJbflE4KLq3ADMT7I/cBywuqo2VdVmYDWwbLp1SZKmbiZHBgcBG4FPJrk1ySeS7AUsqqoH2jYPAova8mLg/oH917e2rbX/nCQrkqxJsmbjxo0zKF2SNGgmYTAPOAw4r6peBPwzT3QJAVBVBdQMXuNJqmpVVS2tqqULFy7cXk8rSXPeTMJgPbC+qm5sjy+jC4eHWvcP7fvDbf0G4ICB/Ze0tq21S5JmybTDoKoeBO5P8vzWdAzwDeBKYGxE0HLgirZ8JXBqG1V0FPBI6066Gjg2yYJ24vjY1iZJmiXzZrj/O4C/TrIbcA/wFrqAuTTJacB9wBvbtlcBJwDrgMfatlTVpiRnAje37d5fVZtmWJckaQpmFAZVdRuwdIJVx0ywbQGnb+V5LgAumEktkqTp8wpkSZJhIEkyDCRJGAaSJAwDSRKGgSQJw0CShGEgScIwkCRhGEiSMAwkSRgGkiQMA0kShoEkCcNAkoRhIEnCMJAkYRhIkjAMJEkYBpIkDANJEoaBJAnDQJKEYSBJwjCQJGEYSJIwDCRJGAaSJAwDSRKGgSQJw0CShGEgScIwkCRhGEiSMAwkSWyHMEiyS5Jbk3yhPT4oyY1J1iX5bJLdWvvu7fG6tv7Aged4X2v/ZpLjZlqTJGlqtseRwbuAuwYefxA4p6qeC2wGTmvtpwGbW/s5bTuSHAqcArwAWAacm2SX7VCXJGmSZhQGSZYArwY+0R4HOBq4rG1yIXBSWz6xPaatP6ZtfyJwSVX9qKruBdYBR8ykLknS1Mz0yOC/A/8J+Fl7vB/wvara0h6vBxa35cXA/QBt/SNt+8fbJ9jnSZKsSLImyZqNGzfOsHRJ0phph0GS1wAPV9Xa7VjPU6qqVVW1tKqWLly4cLZeVpJ2evNmsO9LgNcmOQF4OvAM4CPA/CTz2qf/JcCGtv0G4ABgfZJ5wD7AdwfaxwzuI0maBdM+Mqiq91XVkqo6kO4E8Feq6neBa4GT22bLgSva8pXtMW39V6qqWvspbbTRQcDBwE3TrUuSNHUzOTLYmvcClyT5AHArcH5rPx+4OMk6YBNdgFBVdya5FPgGsAU4vap+ugPqkiRtxXYJg6q6DriuLd/DBKOBqupfgDdsZf+zgLO2Ry2SpKnzCmRJkmEgSTIMJEkYBpIkDANJEoaBJAnDQJKEYSBJwjCQJGEYSJIwDCRJGAaSJAwDSRKGgSQJw0CShGEgScIwkCRhGEiSMAwkSRgGkiQMA0kShoEkCcNAkoRhIEnCMJAkYRhIkjAMJEkYBpIkDANJEoaBJAnDQJKEYSBJwjCQJGEYSJKYQRgkOSDJtUm+keTOJO9q7fsmWZ3k7vZ9QWtPko8mWZfk9iSHDTzX8rb93UmWz/zHkiRNxUyODLYA/7GqDgWOAk5PcihwBnBNVR0MXNMeAxwPHNy+VgDnQRcewErgSOAIYOVYgEiSZse0w6CqHqiqW9ry94G7gMXAicCFbbMLgZPa8onARdW5AZifZH/gOGB1VW2qqs3AamDZdOuSJE3ddjlnkORA4EXAjcCiqnqgrXoQWNSWFwP3D+y2vrVtrX2i11mRZE2SNRs3btwepUuS2A5hkGRv4G+BP6yqRwfXVVUBNdPXGHi+VVW1tKqWLly4cHs9rSTNeTMKgyS70gXBX1fV5a35odb9Q/v+cGvfABwwsPuS1ra1dknSLJnJaKIA5wN3VdWfD6y6EhgbEbQcuGKg/dQ2qugo4JHWnXQ1cGySBe3E8bGtTZI0S+bNYN+XAG8G7khyW2v7z8DZwKVJTgPuA97Y1l0FnACsAx4D3gJQVZuSnAnc3LZ7f1VtmkFdkqQpmnYYVNU/ANnK6mMm2L6A07fyXBcAF0y3FknSzHgFsiTJMJAkGQaSJAwDSRKGgSQJw0CShGEgScIwkCRhGEiSMAwkSRgGkiQMA0kShoEkCcNAkoRhIEnCMJAkYRhIkjAMJEkYBpIkDANJEoaBJAnDQJKEYSBJwjCQJGEYSJIwDCRJGAaSJAwDSRKGgSQJw0CShGEgScIwkCRhGEiSMAwkSQxRGCRZluSbSdYlOaPveiRpLhmKMEiyC/Bx4HjgUOBNSQ7ttypJmjuGIgyAI4B1VXVPVf0YuAQ4seeaJGnOGJYwWAzcP/B4fWuTJM2CVFXfNZDkZGBZVb21PX4zcGRVvX3cdiuAFe3h84Fv7qCSngn80w567tlg/f2y/n6Ncv07uvZnV9XCiVbM24EvOhUbgAMGHi9pbU9SVauAVTu6mCRrqmrpjn6dHcX6+2X9/Rrl+vusfVi6iW4GDk5yUJLdgFOAK3uuSZLmjKE4MqiqLUneDlwN7AJcUFV39lyWJM0ZQxEGAFV1FXBV33U0O7wragez/n5Zf79Guf7eah+KE8iSpH4NyzkDSVKPDANJkmEgSRqiE8jSmCT7VtWmvuuYK5Is4okr/jdU1UN91qN+zPkjgySXJ/m9JHv3Xcv2kmTfvmuYrCR/PLB8aJJvAWuTfDvJkT2WNilJfn9geUmSa5J8L8lXkzyvz9q2JckLk9wAXAf8Wfu6PskNSQ7rtbhpSLJ3ksOSzO+7lulIsqDP15/zYQAcCZwEfCfJpUle1y58Gwmj/mYK/M7A8n8D3lVVBwFvBM7pp6QpGZwy5c+BzwL70v0s5/VS0eR9iu7f+1er6pXt6xDgD4FP9lvatiU5d2D5pcA3gA8DdyQ5obfCpu+aPl/cMICHq+pk4EDg74C3ARuSfDLJsb1WNjmj/mY66FlV9SWAqroJ2KPneqbqeVW1qqp+VlWfowuFYbZXVd04vrGqbgD26qGeqTpqYPlM4KSq+jfAy4H391PSjKTPF/ecARRAVT0KXAxcnGQ/4A3AGcD/6rG2qXrSm2mSUXgz/ZUkV9L9ISxJsmdVPdbW7dpjXZO1JMlH6epfmGTXqvpJWzfs9X8pyReBi3hi1uADgFOBL/dW1fQ8o6puAaiqe5KMxAfdJKeOLQILBh5TVRfNZi2GAfxgfENVfRf4i/Y17Eb9zXT8fSueBo+f1Bz2bhaA9wwsrwH2BjYn+SWGfH6tqnpnkuPp/g8eP4EMfLzNCDDsDklyO93v/oFJFlTV5hYEo9LVe9DA8u50PRShfUidTV6BPOKSvHxc09qq+kF7Mz25qj7eR13Sjpbk2eOaHqiqHyd5JvCyqrq8j7qmK8ktVdXbiXvD4CkkeVVVre67jp1Z+wS9EvgZ8CfAO4DXA3fRnf94oMfypiXJt6pqqEcSAST5jaq6vS3vCryX7q6DXwc+MHCEqVmQ5NaqelFfrz8S/Wo9Or/vArYlyS5J/iDJmUleMm7dH29tvyHyKbpRIPcD1wI/BE4A/jcj0E2X5PtJHm3fv5/k+8Bzxtr7rm8bPjWwfDbwXLrROHswGv/2ywaW5yc5P8ntSf5nOzIeNW/u88Xn/JFB62+fcBVwdFUN9aiKJJ8A9gRuovtlur6q/qit6/WwczIGPw0l+U5V/fLAutuq6oX9Vbdt7eTxfOA9YxdrJbm3jegaauP+7W8DfquqfpIkwNeq6jf6rfCpDf5+t7+DB4G/ohth9/KqOqnP+kaNJ5DhXwO/x8+fSA7dIfOwO2LsjzbJx4Bzk1wOvImeh6pN0uDR6fjRE0N/5NpOwh4OfCbJ54GP0cPJv2naJ8nr6P6ddx8bBVVVlWRUfoYxSwc+OJyTZHmv1UxCkmVV9eW2vA/ddSq/RddN9+7ZvhLcMIAbgMeq6vrxK5LsqHssb0+Pj5qoqi3AiiR/AnyFbmTLsLsiyd5V9YOqGryA7rnAt3qsa9Kqam2SV9JdgHY98PSeS5qs64HXtuUbkiyqqofaeZxRuIfwLyb5I7oPPc9Iknqiq2PoP0gAf8oTQ3g/DDwA/Dbdkc1f0l0MO2vmfDfRqEvyaeDTY58wBtrfCpxXVaMwvHSnkWR/4EUjMjRzpCVZOa7p3Kra2MLsz6rq1In2Gxbjurme1CXaRxepYTCBJK+pqi/0XcdcleQLVfWavuuYriSrqmpF33VMxyjXPmqSrKfrGgpwOvCcsSObJLfP9jmbUTiU6sMoXsr+uCSjfNs/eOICqFG1tO8CZmCUayfJKH2I+yvgF+i6cy8EngmPD7e+bbaL8ZzBxEbhxOtTGek/aODWvguYoYf7LmAGRrl2GKEPElX1XwcfJ3lpkjcDX++ji8swAJIcwpMvyb88ya9W1V09ljUTo/4H/Z5tbzK8qmrZtrcaTqNcezMyHySS3FRVR7Tlt9INQPgcsDLJYVV19mzWM+e7iZK8F7iE7mjgpvb1E7qhgmf0Wdt0jdIfdJKz2/QBJFma5B66kS33TTDVxtBJ8vaB+p+b5O+TbE5yY5Jf77u+pzLKtW9NVf3+trcaGoODO/4AeFU7WjgW+N1Zr6aq5vQX3fDFXSdo3w24u+/6JlH/Urordz9NN+PkauB7wM10o1p6r3Eb9d8xsHwt3YVPAM8D1vRd3yTqv3Ng+YvA69ryK4D/03d9O2vtk/jZvtR3DZOo8WvAAmC/8b/rwK2zXY/dRN2cOM8C7hvXvn9bN+zOpZvbZz7wVbqLVV6V5Ji27sV9FjcJ85LMq+4aiT2q6maAqvpWkt17rm0yBv+GfrG6+xhQVdcl+YWeapqsUa6dbP1ubAGG+sr1Zh9gLW2W0iT7V9UD6e66OOvnLef80NI2v8nHgLt5Yk73X6abp+XtNW78/rDZxnQOvU58NRlJ3kF3oc3ZwMvoPildDhwN/EpV9Tpfy7YkOYvuXNP7gVOAx+j6fY8GXl9DPER2lGsHSPJTugvnJnrjPKqqRuF+Hj8nyZ7Aoqq6d1Zfd66HAUCb//wInjyn+81V9dP+qpqcJP+X7shgH+BDdDN9fr71t3+4qoZ+ZFGSVwD/nq5raB5dKH8e+GQ9caOYoZXk39LV/xy6OenH6v9gVT3SY2nblOQtwL9jNGv/Ol3X1t0TrLu/qg7ooayRZRiMuCS/SXcj858B76Z7U1pOF2hvq6qv9ljepLTRXIuBG6vqBwPty4b9yAwgyRF0U/rcnOQFwDLgrhrBq5CTXDzsR2NjkpxMd87p56aNSXJSVX2+h7JGlmGwE0vylqoa6hubJ3kn3dWXd9H1876rqq5o60Zh1tWVwPF0RzSr6Y4wrwNeBVxdVWf1V91T28qMvUfTzWtFVb12gvVDK8lLafdjqKpRul3tUDAMdmLjzyEMoyR3AC+u7u5sBwKXARdX1UdG5JzHHXQhtjvdFMpLqurRdPefvrGGeBroJLfQ3UviE3QzrQb4DN35A2qCyRuHybhx+m+j+1DxObqhmX9XszxOf9Q5mmjEpbsH7ISrgFG4wcfTxrqGqurb7fzBZeluaTgKV4JvaeeWHkvy/6rqUYCq+mGSYR+NthR4F/Bf6O7HcFuSHw57CAwYHKe/gm6c/sYkH6KbjdgwmALDYPQtAo4DNo9rD91Q02H3UJIXVtVtAO0I4TXABcAoXPj04yR7VneLyMPHGtv89EMdBlX1M7q5//+mfX+I0XpPeFqSBXQXz6aqNgJU1T8n2dJvaaNnlP7jNbEvAHuPvZkOSnLd7JczZacCT/rDbdccnJrkL/spaUpeVlU/gsffXMfsSncif+hV1XrgDUleDQz7rToHDdU4/VHnOQNJO5W+xumPOsNAkuREdZIkw0CShGEgTVmSk5JUu3Ja2ikYBtLUvQn4h/Zd2ikYBtIUtGGLLwVOo12pm+RpSc5N8o9JVie5qs2bQ5LDk1yfZG2Sq5Ps32P50lYZBtLUnAh8uaq+BXw3yeHA7wAHAocCb6bdQyLJrsD/AE6uqsPpLqQb2rmKNLd50Zk0NW8CPtKWL2mP5wF/0y46ezDJtW3984FfA1YnAdgFeGB2y5UmxzCQJinJvnSzev56kqJ7cy+6ydEm3IXu1pLDfrc5yW4iaQpOpptR9dlVdWC7ecq9wCbg9e3cwSK6ewgDfBNYmOTxbqN2vwNp6BgG0uS9iZ8/Cvhb4JeA9XTTQX8auAV4pKp+TBcgH0zyNeA24F/NXrnS5DkdhbQdJNm7zbi6H3AT8JKqerDvuqTJ8pyBtH18Icl8YDfgTINAo8YjA0mS5wwkSYaBJAnDQJKEYSBJwjCQJAH/H93SdZijxcTpAAAAAElFTkSuQmCC\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.countplot(data['Age'])\n",
        "plt.title('Distribution of Age')\n",
        "plt.xlabel('Different Categories of Age')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 350
        },
        "id": "iwyXobWR_r9G",
        "outputId": "bd239716-9497-4d6b-ff92-a71e0487987c"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.groupby(\"Age\").sum()['Purchase'].plot(kind=\"bar\")\n",
        "plt.title(\"Age and Purchase Analysis\")\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 314
        },
        "id": "rPZhBD33_-u-",
        "outputId": "e87fd92a-6d25-49bc-d6d3-898046094bbe"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(18,5))\n",
        "sns.countplot(data['Product_Category_1'])\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 390
        },
        "id": "YZ1n0u13ABfg",
        "outputId": "b986d9b5-9a3a-41eb-fe2a-51c0445fa150"
      },
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1296x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.groupby('Product_Category_1').mean()['Purchase'].plot(kind='bar',figsize=(18,5))\n",
        "plt.title(\"Product_Category_1 and Purchase Mean Analysis\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 353
        },
        "id": "jXtQQtCBAD-B",
        "outputId": "d2b1c19c-fb2f-410c-f2b7-3f0e0767cef7"
      },
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1296x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.groupby('Product_Category_1').sum()['Purchase'].plot(kind='bar',figsize=(18,5))\n",
        "plt.title(\"Product_Category_1 and Purchase Analysis\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 353
        },
        "id": "v8UIxt5DAGR6",
        "outputId": "9e6605ff-87ea-4e69-9a0e-497e1f9ed788"
      },
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1296x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(18,5))\n",
        "sns.countplot(data['Product_Category_2'])\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 390
        },
        "id": "Vz2vvhQ6AJRJ",
        "outputId": "408b7e38-9ed1-4bce-e4bd-2634feb21209"
      },
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1296x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.corr()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 269
        },
        "id": "l0VFV8xvAN33",
        "outputId": "aba0f619-fd80-482e-d7d7-732e1f54c3b6"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                     User_ID  Occupation  Marital_Status  Product_Category_1  \\\n",
              "User_ID             1.000000   -0.023971        0.020443            0.003825   \n",
              "Occupation         -0.023971    1.000000        0.024280           -0.007618   \n",
              "Marital_Status      0.020443    0.024280        1.000000            0.019888   \n",
              "Product_Category_1  0.003825   -0.007618        0.019888            1.000000   \n",
              "Product_Category_2  0.001529   -0.000384        0.015138            0.540583   \n",
              "Product_Category_3  0.003419    0.013263        0.019473            0.229678   \n",
              "Purchase            0.004716    0.020833       -0.000463           -0.343703   \n",
              "\n",
              "                    Product_Category_2  Product_Category_3  Purchase  \n",
              "User_ID                       0.001529            0.003419  0.004716  \n",
              "Occupation                   -0.000384            0.013263  0.020833  \n",
              "Marital_Status                0.015138            0.019473 -0.000463  \n",
              "Product_Category_1            0.540583            0.229678 -0.343703  \n",
              "Product_Category_2            1.000000            0.543649 -0.209918  \n",
              "Product_Category_3            0.543649            1.000000 -0.022006  \n",
              "Purchase                     -0.209918           -0.022006  1.000000  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-9e288c5f-5100-4cd1-a3cd-399144acfea4\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>User_ID</th>\n",
              "      <th>Occupation</th>\n",
              "      <th>Marital_Status</th>\n",
              "      <th>Product_Category_1</th>\n",
              "      <th>Product_Category_2</th>\n",
              "      <th>Product_Category_3</th>\n",
              "      <th>Purchase</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>User_ID</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>-0.023971</td>\n",
              "      <td>0.020443</td>\n",
              "      <td>0.003825</td>\n",
              "      <td>0.001529</td>\n",
              "      <td>0.003419</td>\n",
              "      <td>0.004716</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Occupation</th>\n",
              "      <td>-0.023971</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.024280</td>\n",
              "      <td>-0.007618</td>\n",
              "      <td>-0.000384</td>\n",
              "      <td>0.013263</td>\n",
              "      <td>0.020833</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Marital_Status</th>\n",
              "      <td>0.020443</td>\n",
              "      <td>0.024280</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.019888</td>\n",
              "      <td>0.015138</td>\n",
              "      <td>0.019473</td>\n",
              "      <td>-0.000463</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Product_Category_1</th>\n",
              "      <td>0.003825</td>\n",
              "      <td>-0.007618</td>\n",
              "      <td>0.019888</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.540583</td>\n",
              "      <td>0.229678</td>\n",
              "      <td>-0.343703</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Product_Category_2</th>\n",
              "      <td>0.001529</td>\n",
              "      <td>-0.000384</td>\n",
              "      <td>0.015138</td>\n",
              "      <td>0.540583</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.543649</td>\n",
              "      <td>-0.209918</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Product_Category_3</th>\n",
              "      <td>0.003419</td>\n",
              "      <td>0.013263</td>\n",
              "      <td>0.019473</td>\n",
              "      <td>0.229678</td>\n",
              "      <td>0.543649</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>-0.022006</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Purchase</th>\n",
              "      <td>0.004716</td>\n",
              "      <td>0.020833</td>\n",
              "      <td>-0.000463</td>\n",
              "      <td>-0.343703</td>\n",
              "      <td>-0.209918</td>\n",
              "      <td>-0.022006</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-9e288c5f-5100-4cd1-a3cd-399144acfea4')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-9e288c5f-5100-4cd1-a3cd-399144acfea4 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-9e288c5f-5100-4cd1-a3cd-399144acfea4');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 34
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.heatmap(data.corr(),annot=True)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 359
        },
        "id": "EuURiDWsAQKf",
        "outputId": "df13a137-5971-49fc-c939-13ae11008f87"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df = data.copy()"
      ],
      "metadata": {
        "id": "CLDVzCctAtDl"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 339
        },
        "id": "o2SIyBORAvVq",
        "outputId": "1fb6ed27-eb3c-4cf8-ccfd-225705f1fa2b"
      },
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   User_ID Product_ID Gender   Age  Occupation City_Category  \\\n",
              "0  1000001  P00069042      F  0-17          10             A   \n",
              "1  1000001  P00248942      F  0-17          10             A   \n",
              "2  1000001  P00087842      F  0-17          10             A   \n",
              "3  1000001  P00085442      F  0-17          10             A   \n",
              "4  1000002  P00285442      M   55+          16             C   \n",
              "\n",
              "  Stay_In_Current_City_Years  Marital_Status  Product_Category_1  \\\n",
              "0                          2               0                   3   \n",
              "1                          2               0                   1   \n",
              "2                          2               0                  12   \n",
              "3                          2               0                  12   \n",
              "4                         4+               0                   8   \n",
              "\n",
              "   Product_Category_2  Product_Category_3  Purchase  \n",
              "0                 NaN                 NaN      8370  \n",
              "1                 6.0                14.0     15200  \n",
              "2                 NaN                 NaN      1422  \n",
              "3                14.0                 NaN      1057  \n",
              "4                 NaN                 NaN      7969  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-2f455aab-3a92-4d51-a470-ad891bcd3437\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>User_ID</th>\n",
              "      <th>Product_ID</th>\n",
              "      <th>Gender</th>\n",
              "      <th>Age</th>\n",
              "      <th>Occupation</th>\n",
              "      <th>City_Category</th>\n",
              "      <th>Stay_In_Current_City_Years</th>\n",
              "      <th>Marital_Status</th>\n",
              "      <th>Product_Category_1</th>\n",
              "      <th>Product_Category_2</th>\n",
              "      <th>Product_Category_3</th>\n",
              "      <th>Purchase</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1000001</td>\n",
              "      <td>P00069042</td>\n",
              "      <td>F</td>\n",
              "      <td>0-17</td>\n",
              "      <td>10</td>\n",
              "      <td>A</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>8370</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1000001</td>\n",
              "      <td>P00248942</td>\n",
              "      <td>F</td>\n",
              "      <td>0-17</td>\n",
              "      <td>10</td>\n",
              "      <td>A</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>6.0</td>\n",
              "      <td>14.0</td>\n",
              "      <td>15200</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1000001</td>\n",
              "      <td>P00087842</td>\n",
              "      <td>F</td>\n",
              "      <td>0-17</td>\n",
              "      <td>10</td>\n",
              "      <td>A</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>12</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1422</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1000001</td>\n",
              "      <td>P00085442</td>\n",
              "      <td>F</td>\n",
              "      <td>0-17</td>\n",
              "      <td>10</td>\n",
              "      <td>A</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>12</td>\n",
              "      <td>14.0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1057</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>1000002</td>\n",
              "      <td>P00285442</td>\n",
              "      <td>M</td>\n",
              "      <td>55+</td>\n",
              "      <td>16</td>\n",
              "      <td>C</td>\n",
              "      <td>4+</td>\n",
              "      <td>0</td>\n",
              "      <td>8</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>7969</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-2f455aab-3a92-4d51-a470-ad891bcd3437')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-2f455aab-3a92-4d51-a470-ad891bcd3437 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-2f455aab-3a92-4d51-a470-ad891bcd3437');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 37
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.get_dummies(df, columns=['Stay_In_Current_City_Years'])"
      ],
      "metadata": {
        "id": "x3aYn933A1Oo"
      },
      "execution_count": 38,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.preprocessing import LabelEncoder\n",
        "lr = LabelEncoder()"
      ],
      "metadata": {
        "id": "KrACoUscA64M"
      },
      "execution_count": 39,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df['Gender'] = lr.fit_transform(df['Gender'])"
      ],
      "metadata": {
        "id": "YuQMqDYyA9Cx"
      },
      "execution_count": 40,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df['Age'] = lr.fit_transform(df['Age'])"
      ],
      "metadata": {
        "id": "0j0OF1PHA_Fx"
      },
      "execution_count": 41,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df['City_Category'] = lr.fit_transform(df['City_Category'])"
      ],
      "metadata": {
        "id": "agSqZSS9BBGA"
      },
      "execution_count": 42,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 270
        },
        "id": "gqfwYwkzBDFC",
        "outputId": "485977f1-fd48-4b7c-fc7b-e202606deff6"
      },
      "execution_count": 43,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   User_ID Product_ID  Gender  Age  Occupation  City_Category  Marital_Status  \\\n",
              "0  1000001  P00069042       0    0          10              0               0   \n",
              "1  1000001  P00248942       0    0          10              0               0   \n",
              "2  1000001  P00087842       0    0          10              0               0   \n",
              "3  1000001  P00085442       0    0          10              0               0   \n",
              "4  1000002  P00285442       1    6          16              2               0   \n",
              "\n",
              "   Product_Category_1  Product_Category_2  Product_Category_3  Purchase  \\\n",
              "0                   3                 NaN                 NaN      8370   \n",
              "1                   1                 6.0                14.0     15200   \n",
              "2                  12                 NaN                 NaN      1422   \n",
              "3                  12                14.0                 NaN      1057   \n",
              "4                   8                 NaN                 NaN      7969   \n",
              "\n",
              "   Stay_In_Current_City_Years_0  Stay_In_Current_City_Years_1  \\\n",
              "0                             0                             0   \n",
              "1                             0                             0   \n",
              "2                             0                             0   \n",
              "3                             0                             0   \n",
              "4                             0                             0   \n",
              "\n",
              "   Stay_In_Current_City_Years_2  Stay_In_Current_City_Years_3  \\\n",
              "0                             1                             0   \n",
              "1                             1                             0   \n",
              "2                             1                             0   \n",
              "3                             1                             0   \n",
              "4                             0                             0   \n",
              "\n",
              "   Stay_In_Current_City_Years_4+  \n",
              "0                              0  \n",
              "1                              0  \n",
              "2                              0  \n",
              "3                              0  \n",
              "4                              1  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-64775166-3524-4db3-a77c-9835981b7b05\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>User_ID</th>\n",
              "      <th>Product_ID</th>\n",
              "      <th>Gender</th>\n",
              "      <th>Age</th>\n",
              "      <th>Occupation</th>\n",
              "      <th>City_Category</th>\n",
              "      <th>Marital_Status</th>\n",
              "      <th>Product_Category_1</th>\n",
              "      <th>Product_Category_2</th>\n",
              "      <th>Product_Category_3</th>\n",
              "      <th>Purchase</th>\n",
              "      <th>Stay_In_Current_City_Years_0</th>\n",
              "      <th>Stay_In_Current_City_Years_1</th>\n",
              "      <th>Stay_In_Current_City_Years_2</th>\n",
              "      <th>Stay_In_Current_City_Years_3</th>\n",
              "      <th>Stay_In_Current_City_Years_4+</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1000001</td>\n",
              "      <td>P00069042</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>10</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>8370</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1000001</td>\n",
              "      <td>P00248942</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>10</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>6.0</td>\n",
              "      <td>14.0</td>\n",
              "      <td>15200</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1000001</td>\n",
              "      <td>P00087842</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>10</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>12</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1422</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1000001</td>\n",
              "      <td>P00085442</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>10</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>12</td>\n",
              "      <td>14.0</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1057</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>1000002</td>\n",
              "      <td>P00285442</td>\n",
              "      <td>1</td>\n",
              "      <td>6</td>\n",
              "      <td>16</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>8</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>7969</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-64775166-3524-4db3-a77c-9835981b7b05')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-64775166-3524-4db3-a77c-9835981b7b05 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-64775166-3524-4db3-a77c-9835981b7b05');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 43
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df['Product_Category_2'] =df['Product_Category_2'].fillna(0).astype('int64')\n",
        "df['Product_Category_3'] =df['Product_Category_3'].fillna(0).astype('int64')"
      ],
      "metadata": {
        "id": "NqVDBzEsBGJG"
      },
      "execution_count": 44,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "r1mvnK4yBIi8",
        "outputId": "1ac2cda2-8c7b-4655-f672-dfb216ccf8b9"
      },
      "execution_count": 45,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "User_ID                          0\n",
              "Product_ID                       0\n",
              "Gender                           0\n",
              "Age                              0\n",
              "Occupation                       0\n",
              "City_Category                    0\n",
              "Marital_Status                   0\n",
              "Product_Category_1               0\n",
              "Product_Category_2               0\n",
              "Product_Category_3               0\n",
              "Purchase                         0\n",
              "Stay_In_Current_City_Years_0     0\n",
              "Stay_In_Current_City_Years_1     0\n",
              "Stay_In_Current_City_Years_2     0\n",
              "Stay_In_Current_City_Years_3     0\n",
              "Stay_In_Current_City_Years_4+    0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 45
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df = df.drop([\"User_ID\",\"Product_ID\"],axis=1)"
      ],
      "metadata": {
        "id": "H4rso_4fBLf9"
      },
      "execution_count": 46,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = df.drop(\"Purchase\",axis=1)"
      ],
      "metadata": {
        "id": "qg6uOpCLBOc7"
      },
      "execution_count": 47,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y=df['Purchase']"
      ],
      "metadata": {
        "id": "Y1DFY_UTBQUG"
      },
      "execution_count": 48,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)"
      ],
      "metadata": {
        "id": "btYXanQgBSiJ"
      },
      "execution_count": 49,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LinearRegression"
      ],
      "metadata": {
        "id": "GhdKIEHjBVdZ"
      },
      "execution_count": 51,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "lr = LinearRegression()\n",
        "lr.fit(X_train,y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TURohB-bBaLd",
        "outputId": "728d57fc-53d4-45dc-86f5-819ac4ca824b"
      },
      "execution_count": 52,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression()"
            ]
          },
          "metadata": {},
          "execution_count": 52
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "lr.intercept_"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rqO4WN1oBdQs",
        "outputId": "4190bc19-4c43-4066-e312-dfe787b9d3ef"
      },
      "execution_count": 53,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "9536.400764131593"
            ]
          },
          "metadata": {},
          "execution_count": 53
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "lr.coef_"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BALPnkRzBfOW",
        "outputId": "1c58bf93-762f-42bc-fb83-fd599865060f"
      },
      "execution_count": 54,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([ 465.82318446,  112.36643445,    5.05508596,  314.06766138,\n",
              "        -58.23217776, -348.4514785 ,   12.98415047,  143.49190467,\n",
              "        -20.83796687,    5.4676518 ,   17.68367185,   -3.96751734,\n",
              "          1.65416056])"
            ]
          },
          "metadata": {},
          "execution_count": 54
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred = lr.predict(X_test)"
      ],
      "metadata": {
        "id": "euZmVp8DBhju"
      },
      "execution_count": 55,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score"
      ],
      "metadata": {
        "id": "MDtyNtYKBj-o"
      },
      "execution_count": 56,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "mean_absolute_error(y_test, y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nbkil8QMBl1b",
        "outputId": "3f88c7ce-732a-4228-90f0-756d80a348ab"
      },
      "execution_count": 57,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "3532.069226165843"
            ]
          },
          "metadata": {},
          "execution_count": 57
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "mean_squared_error(y_test, y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vchHvHyFBn4-",
        "outputId": "999be140-2397-426f-f507-d3761d0c5a40"
      },
      "execution_count": 58,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "21397853.26940751"
            ]
          },
          "metadata": {},
          "execution_count": 58
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "r2_score(y_test, y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ucONiG-fBp7q",
        "outputId": "c585fc47-d7d0-4b94-bbf9-569d766b3638"
      },
      "execution_count": 59,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.15192944521481688"
            ]
          },
          "metadata": {},
          "execution_count": 59
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "tod8ZQCXBqkp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.tree import DecisionTreeRegressor\n",
        "\n",
        "# create a regressor object \n",
        "regressor = DecisionTreeRegressor(random_state = 0)  \n",
        "regressor.fit(X_train, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MKoVnuAjBumg",
        "outputId": "6f9e83a2-15aa-4826-fc8e-63e3276140f3"
      },
      "execution_count": 60,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DecisionTreeRegressor(random_state=0)"
            ]
          },
          "metadata": {},
          "execution_count": 60
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "dt_y_pred = regressor.predict(X_test)"
      ],
      "metadata": {
        "id": "df4ItVlvB1vU"
      },
      "execution_count": 61,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "mean_absolute_error(y_test, dt_y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dvs4kiXLB3xr",
        "outputId": "e435b716-dc80-4944-c566-8adf7a3e4f2e"
      },
      "execution_count": 62,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "2372.0357559134654"
            ]
          },
          "metadata": {},
          "execution_count": 62
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "mean_squared_error(y_test, dt_y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7W21f5eDB5rT",
        "outputId": "aa2580ca-7d1b-462c-b022-9fe6700c2f50"
      },
      "execution_count": 63,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "11300579.466797074"
            ]
          },
          "metadata": {},
          "execution_count": 63
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "r2_score(y_test, dt_y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3wI9pLZBB7rB",
        "outputId": "75a736b2-7a35-4134-957f-bf9e4319d9a0"
      },
      "execution_count": 64,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.5521191505924365"
            ]
          },
          "metadata": {},
          "execution_count": 64
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from math import sqrt\n",
        "print(\"RMSE of Linear Regression Model is \",sqrt(mean_squared_error(y_test, dt_y_pred)))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OPfp09-EB-cp",
        "outputId": "251b357c-6224-4914-e304-d6e637482625"
      },
      "execution_count": 65,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "RMSE of Linear Regression Model is  3361.633452177241\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.ensemble import RandomForestRegressor\n",
        "\n",
        "# create a regressor object \n",
        "RFregressor = RandomForestRegressor(random_state = 0)  \n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "23rLmCXbCDRj",
        "outputId": "7ab214c0-7edc-43af-a71a-b2bc2f609969"
      },
      "execution_count": 67,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "RandomForestRegressor(random_state=0)"
            ]
          },
          "metadata": {},
          "execution_count": 67
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "CTJxFudZCdai"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "rf_y_pred = RFregressor.predict(X_test)"
      ],
      "metadata": {
        "id": "DixdkvSgCmWm"
      },
      "execution_count": 68,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "mean_absolute_error(y_test, rf_y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "78VpA0J5Cqbi",
        "outputId": "48cba712-25da-4e53-b851-de0c7464276a"
      },
      "execution_count": 69,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "2222.049109204734"
            ]
          },
          "metadata": {},
          "execution_count": 69
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "mean_squared_error(y_test, rf_y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4du1OlnWCsSS",
        "outputId": "ef99aca7-d7d2-421a-ab33-bdde89aa05c6"
      },
      "execution_count": 70,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "9310769.87311957"
            ]
          },
          "metadata": {},
          "execution_count": 70
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "r2_score(y_test, rf_y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UwGHBiMmCuXM",
        "outputId": "4ceed3dc-6a7e-4789-eb1e-1b0591e580c4"
      },
      "execution_count": 71,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.6309821516972987"
            ]
          },
          "metadata": {},
          "execution_count": 71
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from math import sqrt\n",
        "print(\"RMSE of Linear Regression Model is \",sqrt(mean_squared_error(y_test, rf_y_pred)))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NziFcuSJCwiC",
        "outputId": "34a0a90b-3d36-4edd-dfb0-0c2e87c748a5"
      },
      "execution_count": 72,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "RMSE of Linear Regression Model is  3051.35541573242\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from xgboost.sklearn import XGBRegressor"
      ],
      "metadata": {
        "id": "JSZCwEuKCzKE"
      },
      "execution_count": 73,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "xgb_reg = XGBRegressor(learning_rate=1.0, max_depth=6, min_child_weight=40, seed=0)\n",
        "\n",
        "xgb_reg.fit(X_train, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5DdCd9e-C1Ut",
        "outputId": "c345f121-7da5-45df-c1c9-52587d35bcf3"
      },
      "execution_count": 76,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[14:34:02] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "XGBRegressor(learning_rate=1.0, max_depth=6, min_child_weight=40, seed=0)"
            ]
          },
          "metadata": {},
          "execution_count": 76
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "xgb_y_pred = xgb_reg.predict(X_test)"
      ],
      "metadata": {
        "id": "ch1VRx_RDJrq"
      },
      "execution_count": 77,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "mean_absolute_error(y_test, xgb_y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-2NL1-dnDyDA",
        "outputId": "8f8d3e5d-5cf1-4aaa-de90-c0b12168aa5f"
      },
      "execution_count": 78,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "2144.8588299087473"
            ]
          },
          "metadata": {},
          "execution_count": 78
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "mean_squared_error(y_test, xgb_y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FbRjeHyLDML3",
        "outputId": "e6a08100-9e4c-4dea-81b8-56b758066bd9"
      },
      "execution_count": 79,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "8268802.185235631"
            ]
          },
          "metadata": {},
          "execution_count": 79
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "r2_score(y_test, xgb_y_pred)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iP1efDNlD2dG",
        "outputId": "789be09e-c24e-4cd9-92f0-65bd49318af8"
      },
      "execution_count": 80,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.6722789165646108"
            ]
          },
          "metadata": {},
          "execution_count": 80
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "9XyPaB32D7CN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from math import sqrt\n",
        "print(\"RMSE of Linear Regression Model is \",sqrt(mean_squared_error(y_test, xgb_y_pred)))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1Hw2ra42D4kN",
        "outputId": "22dcad13-fab2-46c6-d17a-85ba60ebd840"
      },
      "execution_count": 81,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "RMSE of Linear Regression Model is  2875.552500865813\n"
          ]
        }
      ]
    }
  ]
}