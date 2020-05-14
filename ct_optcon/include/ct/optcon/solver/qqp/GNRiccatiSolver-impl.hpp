/**********************************************************************************************************************
This file is part of the Control Toolbox (https://github.com/ethz-adrl/control-toolbox), copyright by ETH Zurich.
Licensed under the BSD-2 license (see LICENSE file in main directory)
 **********************************************************************************************************************/

#pragma once

namespace ct {
namespace optcon {


template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR>
GNRiccatiSolverQQ<STATE_DIM, CONTROL_DIM, SCALAR>::GNRiccatiSolverQQ(const std::shared_ptr<QQOCProblem_t>& qqocProblem)
    : QQOCSolver<STATE_DIM, CONTROL_DIM, SCALAR>(qqocProblem), N_(-1)
{
    Eigen::initParallel();
    Eigen::setNbThreads(settings_.nThreadsEigen);
}


template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR>
GNRiccatiSolverQQ<STATE_DIM, CONTROL_DIM, SCALAR>::GNRiccatiSolverQQ(int N)
{
    changeNumberOfStages(N);
}

template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR>
void GNRiccatiSolverQQ<STATE_DIM, CONTROL_DIM, SCALAR>::solve()
{
    for (int i = this->qqocProblem_->getNumberOfStages() - 1; i >= 0; i--)
        solveSingleStage(i);
}


template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR>
void GNRiccatiSolverQQ<STATE_DIM, CONTROL_DIM, SCALAR>::solveSingleStage(int N)
{
    if (N == this->qqocProblem_->getNumberOfStages() - 1)
        initializeCostToGo();

    designController(N);

    if (N > 0)
        computeCostToGo(N);
}


template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR>
void GNRiccatiSolverQQ<STATE_DIM, CONTROL_DIM, SCALAR>::configure(const NLOptConSettings& settings)
{
    settings_ = settings;
    H_corrFix_ = settings_.epsilon * ControlMatrix::Identity();
}

template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR>
void GNRiccatiSolverQQ<STATE_DIM, CONTROL_DIM, SCALAR>::computeStatesAndControls()
{
    QQOCProblem_t& p = *this->qqocProblem_;

    this->x_sol_[0].setZero();  // should always be zero (fixed init state)

    for (int k = 0; k < this->qqocProblem_->getNumberOfStages(); k++)
    {
        //! control update rule in diff coordinates
        this->u_sol_[k] = this->lv_[k] + this->L_[k] * this->x_sol_[k];

        //! state update rule in diff coordinates
        this->x_sol_[k + 1] = p.fx_[k] * this->x_sol_[k] + p.fu_[k] * (this->u_sol_[k]) + p.fo_[k];
    }
}

template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR>
void GNRiccatiSolverQQ<STATE_DIM, CONTROL_DIM, SCALAR>::computeFeedbackMatrices()
{ /*no action required, already computed in backward pass */
}

template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR>
void GNRiccatiSolverQQ<STATE_DIM, CONTROL_DIM, SCALAR>::compute_lv()
{ /*no action required, already computed in backward pass*/
}

template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR>
SCALAR GNRiccatiSolverQQ<STATE_DIM, CONTROL_DIM, SCALAR>::getSmallestEigenvalue()
{
    return smallestEigenvalue_;
}

template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR>
void GNRiccatiSolverQQ<STATE_DIM, CONTROL_DIM, SCALAR>::setProblemImpl(std::shared_ptr<QQOCProblem_t> qqocProblem)
{
    if (qqocProblem->isConstrained())
    {
        throw std::runtime_error(
            "Selected wrong solver - GNRiccatiSolverQQ cannot handle constrained problems. Use a different solver");
    }

    const int& N = qqocProblem->getNumberOfStages();
    changeNumberOfStages(N);
}


template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR>
void GNRiccatiSolverQQ<STATE_DIM, CONTROL_DIM, SCALAR>::changeNumberOfStages(int N)
{
    if (N <= 0)
        return;

    if (N_ == N)
        return;

    gv_.resize(N);
    G_.resize(N);

    H_.resize(N);
    Hi_.resize(N);
    Hi_inverse_.resize(N);

    this->lv_.resize(N);
    this->L_.resize(N);

    this->x_sol_.resize(N + 1);
    this->u_sol_.resize(N);

    sv_.resize(N + 1);
    S_.resize(N + 1);

    N_ = N;
}


template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR>
void GNRiccatiSolverQQ<STATE_DIM, CONTROL_DIM, SCALAR>::initializeCostToGo()
{
    //! since intializeCostToGo is the first call, we initialize the smallestEigenvalue here.
    smallestEigenvalue_ = std::numeric_limits<SCALAR>::infinity();

    // initialize quadratic approximation of cost to go
    const int& N = this->qqocProblem_->getNumberOfStages();
    QQOCProblem_t& p = *this->qqocProblem_;

    S_[N] = p.Q_[N];
    sv_[N] = p.qv_[N];
}


template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR>
void GNRiccatiSolverQQ<STATE_DIM, CONTROL_DIM, SCALAR>::computeCostToGo(size_t k)
{
    QQOCProblem_t& p = *this->qqocProblem_;

    S_[k] = p.Q_[k];
    S_[k].noalias() += p.fx_[k].transpose() * S_[k + 1] * p.fx_[k];
    S_[k].noalias() -= this->L_[k].transpose() * Hi_[k] * this->L_[k];

    S_[k] = 0.5 * (S_[k] + S_[k].transpose()).eval();

    sv_[k] = p.qv_[k];
    sv_[k].noalias() += p.fx_[k].transpose() * sv_[k + 1];
    sv_[k].noalias() += p.fx_[k].transpose() * S_[k + 1] * p.fo_[k];
    sv_[k].noalias() += this->L_[k].transpose() * Hi_[k] * this->lv_[k];
    sv_[k].noalias() += this->L_[k].transpose() * gv_[k];
    sv_[k].noalias() += G_[k].transpose() * this->lv_[k];
}


template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR>
void GNRiccatiSolverQQ<STATE_DIM, CONTROL_DIM, SCALAR>::designController(size_t k)
{
    QQOCProblem_t& p = *this->qqocProblem_;

    gv_[k] = p.rv_[k];
    gv_[k].noalias() += p.fu_[k].transpose() * sv_[k + 1];
    gv_[k].noalias() += p.fu_[k].transpose() * S_[k + 1].template selfadjointView<Eigen::Lower>() * p.fo_[k];

    G_[k] = p.P_[k];
    //G_[k].noalias() += fu_[k].transpose() * S_[k+1] * fx_[k];
    G_[k].noalias() += p.fu_[k].transpose() * S_[k + 1].template selfadjointView<Eigen::Lower>() * p.fx_[k];

    H_[k] = p.R_[k];
    //H_[k].noalias() += fu_[k].transpose() * S_[k+1] * fu_[k];
    H_[k].noalias() += p.fu_[k].transpose() * S_[k + 1].template selfadjointView<Eigen::Lower>() * p.fu_[k];

    if (settings_.fixedHessianCorrection)
    {
        if (settings_.epsilon > 1e-10)
            Hi_[k] = H_[k] + settings_.epsilon * ControlMatrix::Identity();
        else
            Hi_[k] = H_[k];

        if (settings_.recordSmallestEigenvalue)
        {
            // compute eigenvalues with eigenvectors enabled
            eigenvalueSolver_.compute(Hi_[k], Eigen::ComputeEigenvectors);
            const ControlMatrix& V = eigenvalueSolver_.eigenvectors().real();
            const ControlVector& lambda = eigenvalueSolver_.eigenvalues();

            smallestEigenvalue_ = std::min(smallestEigenvalue_, lambda.minCoeff());

            // Corrected Eigenvalue Matrix
            ControlMatrix D = ControlMatrix::Zero();
            // make D positive semi-definite (as described in IV. B.)
            D.diagonal() = lambda.cwiseMax(settings_.epsilon);

            // reconstruct H
            ControlMatrix Hi_regular = V * D * V.transpose();

            // invert D
            ControlMatrix D_inverse = ControlMatrix::Zero();
            // eigenvalue-wise inversion
            D_inverse.diagonal() = -1.0 * D.diagonal().cwiseInverse();
            ControlMatrix Hi_inverse_regular = V * D_inverse * V.transpose();

            if (!Hi_inverse_[k].isApprox(Hi_inverse_regular, 1e-4))
            {
                std::cout << "warning, inverses not identical at " << k << std::endl;
                std::cout << "Hi_inverse_fixed - Hi_inverse_regular: " << std::endl
                          << Hi_inverse_[k] - Hi_inverse_regular << std::endl
                          << std::endl;
            }
        }

        Hi_inverse_[k] = -Hi_[k].template selfadjointView<Eigen::Lower>().llt().solve(ControlMatrix::Identity());

        // calculate FB gain update
        this->L_[k].noalias() = Hi_inverse_[k].template selfadjointView<Eigen::Lower>() * G_[k];

        // calculate FF update
        this->lv_[k].noalias() = Hi_inverse_[k].template selfadjointView<Eigen::Lower>() * gv_[k];
    }
    else
    {
        // compute eigenvalues with eigenvectors enabled
        eigenvalueSolver_.compute(H_[k], Eigen::ComputeEigenvectors);
        const ControlMatrix& V = eigenvalueSolver_.eigenvectors().real();
        const ControlVector& lambda = eigenvalueSolver_.eigenvalues();

        if (settings_.recordSmallestEigenvalue)
        {
            smallestEigenvalue_ = std::min(smallestEigenvalue_, lambda.minCoeff());
        }

        // Corrected Eigenvalue Matrix
        ControlMatrix D = ControlMatrix::Zero();
        // make D positive semi-definite (as described in IV. B.)
        D.diagonal() = lambda.cwiseMax(settings_.epsilon);

        // reconstruct H
        Hi_[k].noalias() = V * D * V.transpose();

        // invert D
        ControlMatrix D_inverse = ControlMatrix::Zero();
        // eigenvalue-wise inversion
        D_inverse.diagonal() = -1.0 * D.diagonal().cwiseInverse();
        Hi_inverse_[k].noalias() = V * D_inverse * V.transpose();

        // calculate FB gain update
        this->L_[k].noalias() = Hi_inverse_[k] * G_[k];

        // calculate FF update
        this->lv_[k].noalias() = Hi_inverse_[k] * gv_[k];
    }
}


template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR>
void GNRiccatiSolverQQ<STATE_DIM, CONTROL_DIM, SCALAR>::logToMatlab()
{
#ifdef MATLAB_FULL_LOG

    matFile_.open("GNRiccatiSolverQQ.mat");

    matFile_.put("sv", sv_.toImplementation());
    matFile_.put("S", S_.toImplementation());
    matFile_.put("L", this->L_.toImplementation());
    matFile_.put("H", H_.toImplementation());
    matFile_.put("Hi_", Hi_.toImplementation());
    matFile_.put("Hi_inverse", Hi_inverse_.toImplementation());
    matFile_.put("G", G_.toImplementation());
    matFile_.put("gv", gv_.toImplementation());

    matFile_.close();
#endif
}

template <size_t STATE_DIM, size_t CONTROL_DIM, typename SCALAR>
void GNRiccatiSolverQQ<STATE_DIM, CONTROL_DIM, SCALAR>::initializeAndAllocate()
{
    // do nothing
}


}  // namespace optcon
}  // namespace ct
